//! NURBS adaptive quadtree tessellation.

use brepkit_math::det_hash::DetHashMap;
use brepkit_math::madd::Madd;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;

use super::{TriangleMesh, TriangleMeshUV};

/// A cell in the adaptive quadtree for NURBS tessellation.
pub(super) struct AdaptiveCell {
    u_min: f64,
    u_max: f64,
    v_min: f64,
    v_max: f64,
    depth: u8,
    /// Indices into the cell vec; `None` means this is a leaf cell.
    children: Option<[usize; 4]>,
}

/// Maximum recursion depth for adaptive subdivision.
const MAX_DEPTH: u8 = 6;

/// Initial grid resolution (cells per direction).
const INITIAL_CELLS: usize = 4;

/// Most crease lines the root grid takes per direction. A surface with more, such
/// as a faceted sheet, keeps the even grid there: each line multiplies the roots
/// the other way, and this many keeps them in the hundreds.
const MAX_CREASE_LINES: usize = 16;

/// Cells a quadtree stops growing at. An even grid's tree cannot reach it, only one
/// on many crease-lined roots.
const MAX_CELLS: usize = 200_000;

/// Compute the v-parameter range for a surface by projecting boundary vertices.
///
/// `project_v` maps a 3D point to its v-parameter on the surface.
/// Falls back to (-1.0, 1.0) if the face has no usable vertices.
pub(super) fn compute_v_param_range(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    project_v: impl Fn(Point3) -> f64,
) -> (f64, f64) {
    let mut v_min = f64::MAX;
    let mut v_max = f64::MIN;

    if let Ok(wire) = topo.wire(face_data.outer_wire()) {
        for oe in wire.edges() {
            if let Ok(edge) = topo.edge(oe.edge()) {
                for &vid in &[edge.start(), edge.end()] {
                    if let Ok(vertex) = topo.vertex(vid) {
                        let v = project_v(vertex.point());
                        v_min = v_min.min(v);
                        v_max = v_max.max(v);
                    }
                }
            }
        }
    }

    if v_min < v_max {
        (v_min, v_max)
    } else {
        (-1.0, 1.0) // fallback
    }
}

/// Compute the tube-angle (v) range for a toroidal face from its wire boundary.
///
/// A full torus has no boundary constraint on v, so the default is the full
/// tube `(0, TAU)`. A toroidal *band* (e.g. a rim-fillet quarter-torus) is
/// bounded by two closed circle edges sitting at distinct constant v; the band
/// fills the arc between them. v is periodic, so two arcs are possible — the
/// fillet band is the shorter one (a 90° rim corner spans π/2; we accept up to
/// just under π). Returns `(0, TAU)` whenever the boundary doesn't clearly
/// describe such a band (preserving full-tube tessellation for every other
/// toroidal face).
pub(super) fn compute_torus_v_range(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    torus: &brepkit_math::surfaces::ToroidalSurface,
) -> (f64, f64) {
    use brepkit_topology::edge::EdgeCurve;
    use std::f64::consts::{PI, TAU};

    // Collect the (constant) v of each closed circular boundary edge.
    let mut circle_vs: Vec<f64> = Vec::new();
    if let Ok(wire) = topo.wire(face_data.outer_wire()) {
        for oe in wire.edges() {
            if let Ok(edge) = topo.edge(oe.edge())
                && matches!(edge.curve(), EdgeCurve::Circle(_))
                && edge.start() == edge.end()
                && let Ok(vertex) = topo.vertex(edge.start())
            {
                circle_vs.push(torus.project_point(vertex.point()).1.rem_euclid(TAU));
            }
        }
    }

    if circle_vs.len() != 2 {
        return (0.0, TAU);
    }
    let (va, vb) = (circle_vs[0], circle_vs[1]);

    // Two candidate arcs between the circles; the band is the shorter one.
    let (lo, hi) = if va <= vb { (va, vb) } else { (vb, va) };
    let forward_span = hi - lo; // arc lo -> hi without wrap
    if forward_span <= PI {
        (lo, hi)
    } else {
        // The wrapped arc hi -> lo + TAU is the shorter one.
        (hi, lo + TAU)
    }
}

/// Compute the v-range (axial extent) for an analytic surface from its face
/// wire boundary vertices.
///
/// Projects all wire vertices onto the surface axis and returns (v_min, v_max).
/// Falls back to (-1.0, 1.0) if the face has no usable vertices.
pub(super) fn compute_axial_range(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    origin: Point3,
    axis: Vec3,
) -> (f64, f64) {
    let mut v_min = f64::MAX;
    let mut v_max = f64::MIN;

    if let Ok(wire) = topo.wire(face_data.outer_wire()) {
        for oe in wire.edges() {
            if let Ok(edge) = topo.edge(oe.edge()) {
                for &vid in &[edge.start(), edge.end()] {
                    if let Ok(vertex) = topo.vertex(vid) {
                        let pt = vertex.point();
                        let to_pt = Vec3::new(
                            pt.x() - origin.x(),
                            pt.y() - origin.y(),
                            pt.z() - origin.z(),
                        );
                        let v = axis.dot(to_pt);
                        v_min = v_min.min(v);
                        v_max = v_max.max(v);
                    }
                }
            }
        }
    }

    if v_min < v_max {
        (v_min, v_max)
    } else {
        (-1.0, 1.0) // fallback
    }
}

/// Compute the angular (u) range for an analytic face from its wire boundary.
///
/// Each edge of the outer wire is walked along its own span, just inside its
/// ends (a pole or apex vertex has no `u` of its own), unwrapped so the walk
/// never jumps a period, and contributes the interval of `u` it covers; the
/// largest gap the intervals leave is the face's opening. Returns `(0, 2*pi)`
/// for a face whose edges cover the full revolution.
pub(super) fn compute_angular_range<F>(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    project: F,
) -> (f64, f64)
where
    F: Fn(Point3) -> (f64, f64),
{
    use brepkit_topology::edge::EdgeCurve;
    use std::f64::consts::{PI, TAU};
    // Samples sit this fraction of a span inside its ends, which leaves
    // slivers of `u` uncovered where edges meet; gaps narrower than
    // `SLIVER` are covered.
    const INSET: f64 = 1e-9;
    const SLIVER: f64 = 1e-7;

    let Ok(wire) = topo.wire(face_data.outer_wire()) else {
        return (0.0, TAU);
    };
    // The walk reads a closed loop; a wire that does not close on itself
    // bounds nothing to read, and keeps the full turn.
    let ends = |oe: &brepkit_topology::wire::OrientedEdge| {
        topo.edge(oe.edge())
            .ok()
            .map(|e| (oe.oriented_start(e), oe.oriented_end(e)))
    };
    let closes = match (
        wire.edges().first().and_then(ends),
        wire.edges().last().and_then(ends),
    ) {
        (Some((start, _)), Some((_, end))) => start == end,
        _ => false,
    };
    if !closes {
        return (0.0, TAU);
    }
    let mut intervals: Vec<(f64, f64)> = Vec::new();
    for oe in wire.edges() {
        let Ok(edge) = topo.edge(oe.edge()) else {
            continue;
        };
        let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) else {
            continue;
        };
        let (sp, ep) = (sv.point(), ev.point());
        let curve = edge.curve();
        if matches!(curve, EdgeCurve::Line) && (ep - sp).length() < 1e-12 {
            continue;
        }
        let (t0, t1) = curve.domain_with_endpoints(sp, ep);
        let pieces: u32 = match curve {
            EdgeCurve::Line => 1,
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => {
                ((t1 - t0).abs() / (PI / 16.0)).ceil().max(1.0) as u32
            }
            EdgeCurve::NurbsCurve(_) => 32,
        };
        let points: Vec<Point3> = (0..=pieces)
            .map(|k| {
                let f = (1.0 - 2.0 * INSET).madd(f64::from(k) / f64::from(pieces), INSET);
                curve.evaluate_with_endpoints((t1 - t0).madd(f, t0), sp, ep)
            })
            .collect();
        // An edge collapsed to one point has no `u` of its own.
        let reach = points
            .iter()
            .fold(0.0_f64, |m, p| m.max((*p - points[0]).length()));
        if reach <= 1e-9 * (sp - Point3::new(0.0, 0.0, 0.0)).length().max(1.0) {
            continue;
        }
        let mut prev: Option<f64> = None;
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for p in points {
            let (mut u, _) = project(p);
            if let Some(p) = prev {
                u -= ((u - p) / TAU).round() * TAU;
            }
            prev = Some(u);
            lo = lo.min(u);
            hi = hi.max(u);
        }
        if hi - lo >= TAU - SLIVER {
            return (0.0, TAU);
        }
        let shift = lo.rem_euclid(TAU) - lo;
        intervals.push((lo + shift, hi + shift));
    }
    if intervals.is_empty() {
        return (0.0, TAU);
    }

    // Union on the circle: split intervals crossing the period end, sort,
    // merge, then find the widest uncovered gap.
    let mut pieces: Vec<(f64, f64)> = Vec::new();
    for (a, b) in intervals {
        if b > TAU {
            pieces.push((a, TAU));
            pieces.push((0.0, b - TAU));
        } else {
            pieces.push((a, b));
        }
    }
    pieces.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut merged: Vec<(f64, f64)> = Vec::new();
    for (a, b) in pieces {
        match merged.last_mut() {
            Some(last) if a <= last.1 + SLIVER => last.1 = last.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    let mut best = (0.0_f64, 0.0_f64, 0.0_f64);
    for (i, &(_, b)) in merged.iter().enumerate() {
        let next = if i + 1 < merged.len() {
            merged[i + 1].0
        } else {
            merged[0].0 + TAU
        };
        if next - b > best.0 {
            best = (next - b, b, next);
        }
    }
    if best.0 <= SLIVER {
        return (0.0, TAU);
    }
    let u_start = best.2.rem_euclid(TAU);
    let u_end = u_start + (TAU - best.0);
    (u_start, u_end)
}

/// Compute the latitude (v) range for a sphere face from its wire boundary.
#[must_use]
pub fn compute_sphere_v_range(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    sphere: &brepkit_math::surfaces::SphericalSurface,
) -> (f64, f64) {
    use std::f64::consts::FRAC_PI_2;

    let mut wire_pts = Vec::new();
    if let Ok(wire) = topo.wire(face_data.outer_wire()) {
        for oe in wire.edges() {
            if let Ok(edge) = topo.edge(oe.edge())
                && let Ok(vertex) = topo.vertex(edge.start())
            {
                wire_pts.push(vertex.point());
            }
        }
    }

    if wire_pts.len() < 3 {
        return (-FRAC_PI_2, FRAC_PI_2);
    }

    let avg_v: f64 = wire_pts
        .iter()
        .map(|pt| sphere.project_point(*pt).1)
        .sum::<f64>()
        / wire_pts.len() as f64;

    let signed_area = projected_signed_area(&wire_pts);
    if signed_area > 0.0 {
        (avg_v, FRAC_PI_2)
    } else {
        (-FRAC_PI_2, avg_v)
    }
}

/// Signed area of a polygon projected onto the XY plane.
/// Positive = CCW winding from +Z, negative = CW.
#[must_use]
pub fn projected_signed_area(pts: &[Point3]) -> f64 {
    let n = pts.len();
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += pts[i].x() * pts[j].y() - pts[j].x() * pts[i].y();
    }
    area * 0.5
}

/// Determine the [`AnalyticKind`] for sphere tessellation based on v-range.
pub(super) fn sphere_analytic_kind(v_range: (f64, f64)) -> super::AnalyticKind {
    use super::AnalyticKind;
    use std::f64::consts::FRAC_PI_2;
    let eps = 1e-6;
    let has_south_pole = (v_range.0 + FRAC_PI_2).abs() < eps;
    let has_north_pole = (v_range.1 - FRAC_PI_2).abs() < eps;
    match (has_south_pole, has_north_pole) {
        (true, true) => AnalyticKind::SpherePole,
        (true, false) => AnalyticKind::ConeApex,
        (false, true) => AnalyticKind::VMaxPole,
        (false, false) => AnalyticKind::General,
    }
}

/// Evaluate the surface normal at `(u, v)`, returning a fallback for degenerate points.
fn safe_normal(surface: &brepkit_math::nurbs::surface::NurbsSurface, u: f64, v: f64) -> Vec3 {
    surface.normal(u, v).unwrap_or(Vec3::new(0.0, 0.0, 1.0))
}

/// Whether a quad cell's normals turn by more than `angular_tol` across any
/// pair of its sampled corners/center.
///
/// `angular_tol <= 0` disables the angular criterion.
#[allow(clippy::similar_names)]
fn cell_exceeds_angular(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    u_min: f64,
    u_max: f64,
    v_min: f64,
    v_max: f64,
    angular_tol: f64,
    creases: &Creases,
) -> bool {
    if angular_tol <= 0.0 {
        return false;
    }
    let u_mid = 0.5 * (u_min + u_max);
    let v_mid = 0.5 * (v_min + v_max);
    // A normal read on a crease is the far side's half the time, so a cell's
    // corners there are read just inside it.
    let (u_min, u_max) = inside_creases(u_min, u_max, &creases.u);
    let (v_min, v_max) = inside_creases(v_min, v_max, &creases.v);
    let normals = [
        safe_normal(surface, u_min, v_min),
        safe_normal(surface, u_max, v_min),
        safe_normal(surface, u_max, v_max),
        safe_normal(surface, u_min, v_max),
        safe_normal(surface, u_mid, v_mid),
    ];
    let mut min_dot = 1.0_f64;
    for i in 0..normals.len() {
        for j in (i + 1)..normals.len() {
            min_dot = min_dot.min(normals[i].dot(normals[j]));
        }
    }
    min_dot.clamp(-1.0, 1.0).acos() > angular_tol
}

/// Compute the refinement error for a quad cell using combined metrics.
#[allow(clippy::similar_names)]
fn cell_refinement_error(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    u_min: f64,
    u_max: f64,
    v_min: f64,
    v_max: f64,
    creases: &Creases,
) -> f64 {
    let u_mid = 0.5 * (u_min + u_max);
    let v_mid = 0.5 * (v_min + v_max);

    let p00 = surface.evaluate(u_min, v_min);
    let p10 = surface.evaluate(u_max, v_min);
    let p11 = surface.evaluate(u_max, v_max);
    let p01 = surface.evaluate(u_min, v_max);
    let p_mid = surface.evaluate(u_mid, v_mid);

    let bilinear_mid = Point3::new(
        0.25 * (p00.x() + p10.x() + p11.x() + p01.x()),
        0.25 * (p00.y() + p10.y() + p11.y() + p01.y()),
        0.25 * (p00.z() + p10.z() + p11.z() + p01.z()),
    );
    let sag = (p_mid - bilinear_mid).length();

    let (u_in_lo, u_in_hi) = inside_creases(u_min, u_max, &creases.u);
    let (v_in_lo, v_in_hi) = inside_creases(v_min, v_max, &creases.v);
    let normals = [
        safe_normal(surface, u_in_lo, v_in_lo),
        safe_normal(surface, u_in_hi, v_in_lo),
        safe_normal(surface, u_in_hi, v_in_hi),
        safe_normal(surface, u_in_lo, v_in_hi),
        safe_normal(surface, u_mid, v_mid),
    ];

    let mut max_normal_dev = 0.0_f64;
    for i in 0..normals.len() {
        for j in (i + 1)..normals.len() {
            let dev = 1.0 - normals[i].dot(normals[j]);
            max_normal_dev = max_normal_dev.max(dev);
        }
    }

    let edge_mids = [
        surface.evaluate(u_mid, v_min),
        surface.evaluate(u_mid, v_max),
        surface.evaluate(u_min, v_mid),
        surface.evaluate(u_max, v_mid),
    ];

    let edge_linear_mids = [
        lerp_point(p00, p10),
        lerp_point(p01, p11),
        lerp_point(p00, p01),
        lerp_point(p10, p11),
    ];

    let mut max_edge_sag = 0.0_f64;
    for i in 0..4 {
        let edge_sag = (edge_mids[i] - edge_linear_mids[i]).length();
        max_edge_sag = max_edge_sag.max(edge_sag);
    }

    let diag = (p11 - p00).length().max((p10 - p01).length());
    let normal_sag = max_normal_dev * diag * 0.5;

    // A twisted cell's middle is its corners' average, so `sag` misses its
    // twist; read the cell against the triangles it splits into instead.
    let probes = PROBES.map(|(a, b)| {
        if (a, b) == (0.5, 0.5) {
            p_mid
        } else {
            surface.evaluate(
                (u_max - u_min).madd(a, u_min),
                (v_max - v_min).madd(b, v_min),
            )
        }
    });
    let twist = split_stray([p00, p10, p01, p11], &probes);

    sag.max(max_edge_sag).max(normal_sag).max(twist)
}

/// Linear interpolation (midpoint) of two points.
fn lerp_point(a: Point3, b: Point3) -> Point3 {
    Point3::new(
        0.5 * (a.x() + b.x()),
        0.5 * (a.y() + b.y()),
        0.5 * (a.z() + b.z()),
    )
}

/// The root cells' lines, laid out in `cells` by `u` span then `v` span.
struct RootGrid {
    u: Vec<f64>,
    v: Vec<f64>,
}

impl RootGrid {
    /// The first root holding `(u, v)`, a point on a shared line reading the lower.
    fn holding(&self, u: f64, v: f64) -> Option<usize> {
        let span = |lines: &[f64], at: f64| {
            let (&first, &last) = (lines.first()?, lines.last()?);
            (at >= first && at <= last)
                .then(|| lines.partition_point(|&line| line < at).saturating_sub(1))
        };
        Some(span(&self.u, u)? * (self.v.len() - 1) + span(&self.v, v)?)
    }
}

/// A surface's creases along knot lines, by parameter.
struct Creases {
    u: Vec<f64>,
    v: Vec<f64>,
}

/// The knot lines a surface actually creases along, in `u` and in `v`.
///
/// A cell straddling a crease has normals that disagree however small it gets,
/// so it would divide to the depth limit all along the line. Where the crease is
/// a cell edge, each side is smooth. A knot repeated up to the degree is no
/// crease by itself: a rational arc's is smooth, and meshes as it did.
fn creases(surface: &brepkit_math::nurbs::surface::NurbsSurface) -> Creases {
    const PROBES: u32 = 5;
    const CREASE_ANGLE: f64 = 1e-3;
    // Each full knot with the distinct knots either side, so a probe can step a
    // share of its own span off the line: a share of the whole domain can pass
    // through a narrow span, or over a second fold close by.
    let lines = |knots: &[f64], degree: usize, lo: f64, hi: f64| -> Vec<(f64, f64, f64)> {
        let runs: Vec<&[f64]> = knots
            .chunk_by(|a, b| (a - b).abs() <= 1e-12 * (hi - lo))
            .collect();
        let mut found = Vec::new();
        for (i, run) in runs.iter().enumerate() {
            if run[0] > lo && run[0] < hi && run.len() >= degree && i > 0 && i + 1 < runs.len() {
                found.push((runs[i - 1][0], run[0], runs[i + 1][0]));
            }
        }
        found
    };
    let (u_lo, u_hi) = surface.domain_u();
    let (v_lo, v_hi) = surface.domain_v();
    let along =
        |k: u32, lo: f64, hi: f64| lo + (hi - lo) * (f64::from(k) + 0.5) / f64::from(PROBES);
    let bends = |a: Vec3, b: Vec3| a.cross(b).length() > CREASE_ANGLE || a.dot(b) < 0.0;
    let share = |width: f64| 1e-6 * width;
    let u_lines = lines(surface.knots_u(), surface.degree_u(), u_lo, u_hi)
        .into_iter()
        .filter(|&(before, u, after)| {
            (0..PROBES).any(|k| {
                let v = along(k, v_lo, v_hi);
                bends(
                    safe_normal(surface, u - share(u - before), v),
                    safe_normal(surface, u + share(after - u), v),
                )
            })
        })
        .map(|(_, u, _)| u)
        .collect();
    let v_lines = lines(surface.knots_v(), surface.degree_v(), v_lo, v_hi)
        .into_iter()
        .filter(|&(before, v, after)| {
            (0..PROBES).any(|k| {
                let u = along(k, u_lo, u_hi);
                bends(
                    safe_normal(surface, u, v - share(v - before)),
                    safe_normal(surface, u, v + share(after - v)),
                )
            })
        })
        .map(|(_, v, _)| v)
        .collect();
    let capped = |lines: Vec<f64>| {
        if lines.len() > MAX_CREASE_LINES {
            Vec::new()
        } else {
            lines
        }
    };
    Creases {
        u: capped(u_lines),
        v: capped(v_lines),
    }
}

/// `[lo, hi]`, each end on a crease moved a hair into the span.
fn inside_creases(lo: f64, hi: f64, creases: &[f64]) -> (f64, f64) {
    let hair = 1e-6 * (hi - lo);
    let on_crease = |at: f64| {
        creases
            .iter()
            .any(|&crease| (crease - at).abs() <= 1e-3 * hair)
    };
    (
        if on_crease(lo) { lo + hair } else { lo },
        if on_crease(hi) { hi - hair } else { hi },
    )
}

/// The even root grid's lines across `[lo, hi]`, with `creases` among them. An even
/// line beside a crease gives way to it, leaving no sliver between the two.
fn root_lines(lo: f64, hi: f64, creases: &[f64]) -> Vec<f64> {
    #[allow(clippy::cast_precision_loss)]
    let cell = (hi - lo) / INITIAL_CELLS as f64;
    #[allow(clippy::cast_precision_loss)]
    let even = |i: usize| lo + (i as f64) * cell;
    if creases.is_empty() {
        return (0..=INITIAL_CELLS).map(even).collect();
    }
    // The ends by value: `lo + 4 * cell` can fall a hair short of `hi`, and a
    // crease near it would then take the face's last strip with it.
    let mut lines = vec![lo, hi];
    lines.extend((1..INITIAL_CELLS).map(even).filter(|&line| {
        creases
            .iter()
            .all(|&crease| (crease - line).abs() > 0.125 * cell)
    }));
    lines.extend_from_slice(creases);
    lines.sort_by(f64::total_cmp);
    lines
}

/// Build the adaptive quadtree by recursive subdivision.
#[allow(clippy::similar_names)]
fn build_quadtree(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    cells: &mut Vec<AdaptiveCell>,
    cell_idx: usize,
    threshold: f64,
    angular_tol: f64,
    creases: &Creases,
) {
    let cell = &cells[cell_idx];
    if cell.depth >= MAX_DEPTH || cells.len() >= MAX_CELLS {
        return;
    }

    let u_min = cell.u_min;
    let u_max = cell.u_max;
    let v_min = cell.v_min;
    let v_max = cell.v_max;
    let depth = cell.depth;

    let error = cell_refinement_error(surface, u_min, u_max, v_min, v_max, creases);
    let angular_exceeded =
        cell_exceeds_angular(surface, u_min, u_max, v_min, v_max, angular_tol, creases);
    if error <= threshold && !angular_exceeded {
        return;
    }

    let u_mid = 0.5 * (u_min + u_max);
    let v_mid = 0.5 * (v_min + v_max);
    let child_depth = depth + 1;

    let c0 = cells.len();
    cells.push(AdaptiveCell {
        u_min,
        u_max: u_mid,
        v_min,
        v_max: v_mid,
        depth: child_depth,
        children: None,
    });
    cells.push(AdaptiveCell {
        u_min: u_mid,
        u_max,
        v_min,
        v_max: v_mid,
        depth: child_depth,
        children: None,
    });
    cells.push(AdaptiveCell {
        u_min,
        u_max: u_mid,
        v_min: v_mid,
        v_max,
        depth: child_depth,
        children: None,
    });
    cells.push(AdaptiveCell {
        u_min: u_mid,
        u_max,
        v_min: v_mid,
        v_max,
        depth: child_depth,
        children: None,
    });

    cells[cell_idx].children = Some([c0, c0 + 1, c0 + 2, c0 + 3]);

    for i in 0..4 {
        build_quadtree(surface, cells, c0 + i, threshold, angular_tol, creases);
    }
}

/// Conforming pass: ensure no more than 1 level difference between adjacent leaf cells.
fn conforming_pass(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    cells: &mut Vec<AdaptiveCell>,
    roots: &RootGrid,
) {
    for _pass in 0..MAX_DEPTH {
        let mut to_subdivide = Vec::new();

        let len = cells.len();
        for i in 0..len {
            if cells[i].children.is_some() {
                continue;
            }

            let depth = cells[i].depth;
            let u_min = cells[i].u_min;
            let u_max = cells[i].u_max;
            let v_min = cells[i].v_min;
            let v_max = cells[i].v_max;

            if needs_conforming_subdivision(cells, roots, depth, u_min, u_max, v_min, v_max) {
                to_subdivide.push(i);
            }
        }

        if to_subdivide.is_empty() {
            break;
        }

        for &cell_idx in &to_subdivide {
            if cells[cell_idx].children.is_some() {
                continue;
            }
            force_subdivide(surface, cells, cell_idx);
        }
    }
}

/// Check if a leaf cell needs conforming subdivision (neighbor is 2+ levels deeper).
#[allow(clippy::similar_names)]
fn needs_conforming_subdivision(
    cells: &[AdaptiveCell],
    roots: &RootGrid,
    depth: u8,
    u_min: f64,
    u_max: f64,
    v_min: f64,
    v_max: f64,
) -> bool {
    let eps = (u_max - u_min) * 0.01;
    let u_mid = 0.5 * (u_min + u_max);
    let v_mid = 0.5 * (v_min + v_max);

    let probes = [
        (u_mid, v_min - eps),
        (u_mid, v_max + eps),
        (u_min - eps, v_mid),
        (u_max + eps, v_mid),
    ];

    for &(pu, pv) in &probes {
        if let Some(neighbor_depth) = find_leaf_depth_at(cells, roots, pu, pv)
            && neighbor_depth > depth + 1
        {
            return true;
        }
    }
    false
}

/// Find the depth of the leaf cell containing the given parameter point.
fn find_leaf_depth_at(cells: &[AdaptiveCell], roots: &RootGrid, u: f64, v: f64) -> Option<u8> {
    let root = roots.holding(u, v)?;
    find_leaf_depth_recursive(cells, root, u, v)
}

/// Recursively find the leaf depth at a given point within a cell subtree.
fn find_leaf_depth_recursive(cells: &[AdaptiveCell], idx: usize, u: f64, v: f64) -> Option<u8> {
    let cell = &cells[idx];
    if u < cell.u_min || u > cell.u_max || v < cell.v_min || v > cell.v_max {
        return None;
    }

    match cell.children {
        None => Some(cell.depth),
        Some(children) => {
            for &child in &children {
                if let Some(d) = find_leaf_depth_recursive(cells, child, u, v) {
                    return Some(d);
                }
            }
            Some(cell.depth + 1)
        }
    }
}

/// Force-subdivide a leaf cell (for conforming pass, no curvature check).
#[allow(clippy::similar_names)]
fn force_subdivide(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    cells: &mut Vec<AdaptiveCell>,
    cell_idx: usize,
) {
    let cell = &cells[cell_idx];
    if cell.depth >= MAX_DEPTH + 2 {
        return;
    }
    let u_min = cell.u_min;
    let u_max = cell.u_max;
    let v_min = cell.v_min;
    let v_max = cell.v_max;
    let child_depth = cell.depth + 1;

    let u_mid = 0.5 * (u_min + u_max);
    let v_mid = 0.5 * (v_min + v_max);

    let c0 = cells.len();
    cells.push(AdaptiveCell {
        u_min,
        u_max: u_mid,
        v_min,
        v_max: v_mid,
        depth: child_depth,
        children: None,
    });
    cells.push(AdaptiveCell {
        u_min: u_mid,
        u_max,
        v_min,
        v_max: v_mid,
        depth: child_depth,
        children: None,
    });
    cells.push(AdaptiveCell {
        u_min,
        u_max: u_mid,
        v_min: v_mid,
        v_max,
        depth: child_depth,
        children: None,
    });
    cells.push(AdaptiveCell {
        u_min: u_mid,
        u_max,
        v_min: v_mid,
        v_max,
        depth: child_depth,
        children: None,
    });

    cells[cell_idx].children = Some([c0, c0 + 1, c0 + 2, c0 + 3]);

    let _ = surface;
}

/// Mesh a NURBS surface that closes on itself in both directions (a torus
/// image) as a structured grid over its whole domain. The adaptive quadtree
/// keeps one-level steps between neighbouring cells, which leave T-junction
/// cracks, and cannot see that the domain's opposite edges meet. Here every
/// row is sampled alike, and the far seam rows copy the near rows' positions,
/// so a caller that welds by position closes both seams exactly.
pub(super) fn tessellate_periodic_nurbs_grid(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    deflection: f64,
    angular_tol: f64,
) -> TriangleMeshUV {
    let (u_lo, u_hi) = surface.domain_u();
    let (v_lo, v_hi) = surface.domain_v();
    // Fewer than three divisions of a closed direction collapse its seam
    // copies onto the cells' other side.
    let (n_u, n_v) = cap_grid(
        iso_divisions(surface, true, deflection, angular_tol).max(3),
        iso_divisions(surface, false, deflection, angular_tol).max(3),
        GRID_MAX_CELLS,
    );
    let (n_u, n_v) = (n_u.max(3), n_v.max(3));
    let (um, vm) = (0.5 * (u_lo + u_hi), 0.5 * (v_lo + v_hi));
    let duv = surface.first_derivatives(um, vm);
    let right_handed = duv.1.cross(duv.2).dot(safe_normal(surface, um, vm)) >= 0.0;

    let mut positions = Vec::with_capacity((n_u + 1) * (n_v + 1));
    let mut normals = Vec::with_capacity(positions.capacity());
    let mut uvs = Vec::with_capacity(positions.capacity());
    for i in 0..=n_u {
        for j in 0..=n_v {
            #[allow(clippy::cast_precision_loss)]
            let (u, v) = (
                u_lo + (u_hi - u_lo) * i as f64 / n_u as f64,
                v_lo + (v_hi - v_lo) * j as f64 / n_v as f64,
            );
            let (pos, nrm) = if i == n_u || j == n_v {
                let k = (i % n_u) * (n_v + 1) + j % n_v;
                (positions[k], normals[k])
            } else {
                (surface.evaluate(u, v), safe_normal(surface, u, v))
            };
            positions.push(pos);
            normals.push(nrm);
            uvs.push([u, v]);
        }
    }
    let mut indices = Vec::with_capacity(n_u * n_v * 6);
    for i in 0..n_u {
        for j in 0..n_v {
            #[allow(clippy::cast_possible_truncation)]
            let at = |a: usize, b: usize| (a * (n_v + 1) + b) as u32;
            let (i00, i10, i11, i01) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
            if right_handed {
                indices.extend([i00, i10, i11, i00, i11, i01]);
            } else {
                indices.extend([i00, i11, i10, i00, i01, i11]);
            }
        }
    }
    TriangleMeshUV {
        mesh: TriangleMesh {
            positions,
            normals,
            indices,
        },
        uvs,
    }
}

/// The most cells a structured NURBS grid may hold; past it the deflection
/// gives way to a bounded mesh.
pub(super) const GRID_MAX_CELLS: usize = 1 << 16;

/// Shrinks a grid's two division counts together, keeping their ratio, until
/// it holds at most `max_cells` cells. The shrunk grid no longer meets the
/// deflection it was sized for, so this says so.
pub(super) fn cap_grid(n_u: usize, n_v: usize, max_cells: usize) -> (usize, usize) {
    let cells = n_u.saturating_mul(n_v);
    if cells <= max_cells {
        return (n_u, n_v);
    }
    log::warn!(
        "NURBS grid of {n_u} x {n_v} cells exceeds {max_cells}; meshing it coarser than the requested deflection"
    );
    #[allow(clippy::cast_precision_loss)]
    let scale = (max_cells as f64 / cells as f64).sqrt();
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let shrink = |n: usize| ((n as f64 * scale).floor() as usize).max(1);
    (shrink(n_u), shrink(n_v))
}

/// Thins grid lines until they hold at most `max_cells` cells: the longer
/// list loses every second inner line. The thinned grid no longer meets the
/// deflection it was sized for, so this says so.
pub(super) fn cap_lines(us: &mut Vec<f64>, vs: &mut Vec<f64>, max_cells: usize) {
    let cells = |us: &[f64], vs: &[f64]| (us.len() - 1).saturating_mul(vs.len() - 1);
    if cells(us, vs) <= max_cells {
        return;
    }
    log::warn!(
        "NURBS grid of {} x {} cells exceeds {max_cells}; meshing it coarser than the requested deflection",
        us.len() - 1,
        vs.len() - 1
    );
    while cells(us, vs) > max_cells {
        let lines = if us.len() >= vs.len() {
            &mut *us
        } else {
            &mut *vs
        };
        let last = lines.len() - 1;
        let mut k = 0;
        lines.retain(|_| {
            k += 1;
            k == 1 || k == last + 1 || k % 2 == 1
        });
    }
}

/// Divisions of one parameter direction that keep every iso-line chord of a
/// few sampled rows within half the deflection (a grid cell's diagonal sags
/// further than its sides) and within the angular tolerance.
fn iso_divisions(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    along_u: bool,
    deflection: f64,
    angular_tol: f64,
) -> usize {
    let ranges = if along_u {
        (surface.domain_u(), surface.domain_v())
    } else {
        (surface.domain_v(), surface.domain_u())
    };
    iso_divisions_over(
        &|u, v| surface.evaluate(u, v),
        &|u, v| safe_normal(surface, u, v),
        along_u,
        ranges,
        deflection,
        angular_tol,
        4096,
    )
}

/// [`iso_divisions`] over a parameter box: `ranges` is the span divided and
/// the span its sampled rows sit across, and `at` / `normal_at` take
/// `(u, v)` (a caller on a periodic surface wraps them). A straight direction
/// (a ruling) needs one division. The normals' turn is weighed across every
/// chord but one that ends on a pole.
#[allow(clippy::too_many_arguments)]
pub(super) fn iso_divisions_over(
    at: &dyn Fn(f64, f64) -> Point3,
    normal_at: &dyn Fn(f64, f64) -> Vec3,
    along_u: bool,
    ranges: ((f64, f64), (f64, f64)),
    deflection: f64,
    angular_tol: f64,
    max_divisions: usize,
) -> usize {
    let mut n = 1;
    while n < max_divisions
        && !iso_chords_hold(at, normal_at, along_u, ranges, deflection, angular_tol, n)
    {
        n *= 2;
    }
    n
}

/// Whether `n` even divisions of one parameter direction keep every
/// iso-line chord of a few sampled rows within half the deflection and the
/// angular tolerance, as [`iso_divisions_over`] asks of the count it takes.
#[allow(clippy::too_many_arguments)]
pub(super) fn iso_chords_hold(
    at: &dyn Fn(f64, f64) -> Point3,
    normal_at: &dyn Fn(f64, f64) -> Vec3,
    along_u: bool,
    ((lo, hi), (o_lo, o_hi)): ((f64, f64), (f64, f64)),
    deflection: f64,
    angular_tol: f64,
    n: usize,
) -> bool {
    const ROWS: usize = 8;
    let point = |t: f64, o: f64| if along_u { at(t, o) } else { at(o, t) };
    let normal = |t: f64, o: f64| {
        if along_u {
            normal_at(t, o)
        } else {
            normal_at(o, t)
        }
    };
    // An end of the span whose cross iso-line collapses to a point is a
    // pole: the surface has no normal there, and the one it reports can
    // point either way.
    let pole = |t: f64| {
        let (a, b, c) = (
            point(t, o_lo),
            point(t, 0.5 * (o_lo + o_hi)),
            point(t, o_hi),
        );
        let reach = a.x().abs().max(a.y().abs()).max(a.z().abs());
        (b - a).length().max((c - a).length()) <= 1e3 * f64::EPSILON * (1.0 + reach)
    };
    let (pole_lo, pole_hi) = (pole(lo), pole(hi));
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
    let fits = |n: usize| {
        (0..ROWS).all(|row| {
            #[allow(clippy::cast_precision_loss)]
            let o = o_lo + (o_hi - o_lo) * (row as f64 + 0.5) / ROWS as f64;
            (0..n).all(|k| {
                #[allow(clippy::cast_precision_loss)]
                let (t0, t1) = (
                    lo + (hi - lo) * k as f64 / n as f64,
                    lo + (hi - lo) * (k + 1) as f64 / n as f64,
                );
                let (p0, p1) = (point(t0, o), point(t1, o));
                let chord_mid = Point3::new(
                    0.5 * (p0.x() + p1.x()),
                    0.5 * (p0.y() + p1.y()),
                    0.5 * (p0.z() + p1.z()),
                );
                // An inflected span can cross its chord at its middle and
                // stand off it either side, so its quarter points count too.
                let sag = [0.25, 0.75]
                    .iter()
                    .map(|&f| off_chord(point(t0 + (t1 - t0) * f, o), p0, p1))
                    .fold((point(0.5 * (t0 + t1), o) - chord_mid).length(), f64::max);
                let at_pole = (k == 0 && pole_lo) || (k + 1 == n && pole_hi);
                let turned = angular_tol > 0.0
                    && !at_pole
                    && normal(t0, o).dot(normal(t1, o)).clamp(-1.0, 1.0).acos() > angular_tol;
                sag <= 0.5 * deflection && !turned
            })
        })
    };
    fits(n)
}

/// The equal parts a span is cut into when it sags `sag_ratio` times its
/// limit and turns `turn_ratio` times its limit: one when it holds both. A
/// sag falls with the square of a span's length and a turn with the length.
/// The count is bounded, so one tight place in a long span does not size the
/// rest of it; each part is read again.
pub(super) fn parts_needed(sag_ratio: f64, turn_ratio: f64) -> usize {
    const MOST: f64 = 8.0;
    const SPARE: f64 = 1.03;
    if sag_ratio <= 1.0 && turn_ratio <= 1.0 {
        return 1;
    }
    let parts = (SPARE * sag_ratio.sqrt())
        .max(SPARE * turn_ratio)
        .ceil()
        .clamp(2.0, MOST);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let parts = parts as usize;
    parts
}

/// Where a span from `a` to `b` is read against its chord: at even
/// fractions, at each knot inside it, and at the quarter points of each
/// piece between those knots. A curve is one polynomial between two knots,
/// and a bump that fills one such piece can lie between the even fractions.
pub(super) fn span_probes(a: f64, b: f64, knots: &[f64]) -> Vec<f64> {
    const FRACTIONS: [f64; 7] = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875];
    let mut probes: Vec<f64> = FRACTIONS.iter().map(|&f| a + (b - a) * f).collect();
    if knots.is_empty() {
        return probes;
    }
    probes.extend_from_slice(knots);
    let (lo, hi) = (a.min(b), a.max(b));
    let ends = std::iter::once(lo)
        .chain(knots.iter().copied())
        .chain(std::iter::once(hi))
        .collect::<Vec<_>>();
    for piece in ends.windows(2) {
        for f in [0.25, 0.5, 0.75] {
            probes.push(piece[0] + (piece[1] - piece[0]) * f);
        }
    }
    probes
}

/// The parameters of one direction's grid lines across a parameter box, both
/// ends included: a span is cut until every iso-line chord of a few sampled
/// rows lies within half the deflection (a grid cell's diagonal sags further
/// than its sides) and turns its normal within the angular tolerance. The
/// spans follow the surface: a straight run takes one and a tight nose many.
///
/// `ranges` is the span divided and the span its rows sit across, `at` /
/// `normal_at` take `(u, v)`, and `knots` are the divided direction's knots,
/// sorted: a span holding many is cut unread, and each one in a span is read.
#[allow(clippy::too_many_arguments)]
pub(super) fn iso_breaks_over(
    at: &dyn Fn(f64, f64) -> Point3,
    normal_at: &dyn Fn(f64, f64) -> Vec3,
    along_u: bool,
    ((lo, hi), (o_lo, o_hi)): ((f64, f64), (f64, f64)),
    knots: &[f64],
    deflection: f64,
    angular_tol: f64,
    max_divisions: usize,
) -> Vec<f64> {
    const ROWS: usize = 8;
    const MAX_DEPTH: usize = 12;
    const MAX_KNOTS: usize = 8;
    let point = |t: f64, o: f64| if along_u { at(t, o) } else { at(o, t) };
    let normal = |t: f64, o: f64| {
        if along_u {
            normal_at(t, o)
        } else {
            normal_at(o, t)
        }
    };
    // An end of the span whose cross iso-line collapses to a point is a
    // pole: the surface has no normal there.
    let pole = |t: f64| {
        let (a, b, c) = (
            point(t, o_lo),
            point(t, 0.5 * (o_lo + o_hi)),
            point(t, o_hi),
        );
        let reach = a.x().abs().max(a.y().abs()).max(a.z().abs());
        (b - a).length().max((c - a).length()) <= 1e3 * f64::EPSILON * (1.0 + reach)
    };
    let (pole_lo, pole_hi) = (pole(lo), pole(hi));
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
    let parts = |a: f64, b: f64| {
        let from = knots.partition_point(|&k| k <= a.min(b));
        let to = knots.partition_point(|&k| k < a.max(b)).max(from);
        if to - from > MAX_KNOTS {
            return 2;
        }
        let probes = span_probes(a, b, &knots[from..to]);
        let (mut sag, mut turn) = (0.0_f64, 0.0_f64);
        for row in 0..ROWS {
            #[allow(clippy::cast_precision_loss)]
            let o = o_lo + (o_hi - o_lo) * (row as f64 + 0.5) / ROWS as f64;
            let (pa, pb) = (point(a, o), point(b, o));
            for &t in &probes {
                sag = sag.max(off_chord(point(t, o), pa, pb));
            }
            if angular_tol > 0.0 {
                let ends = [
                    if a.to_bits() == lo.to_bits() && pole_lo {
                        0.25
                    } else {
                        0.0
                    },
                    0.25,
                    0.5,
                    0.75,
                    if b.to_bits() == hi.to_bits() && pole_hi {
                        0.75
                    } else {
                        1.0
                    },
                ];
                let mut turned = 0.0;
                let mut last = normal(a + (b - a) * ends[0], o);
                for f in &ends[1..] {
                    let next = normal(a + (b - a) * f, o);
                    turned += last.dot(next).clamp(-1.0, 1.0).acos();
                    last = next;
                }
                turn = turn.max(turned);
            }
        }
        let turn_ratio = if angular_tol > 0.0 {
            turn / angular_tol
        } else {
            0.0
        };
        parts_needed(sag / (0.5 * deflection), turn_ratio)
    };

    // Level by level, so a budget that runs out leaves every part of the
    // span equally read, not its start fine and its end whole.
    let mut spans = vec![(lo, hi, false)];
    for _ in 0..MAX_DEPTH {
        let mut next = Vec::with_capacity(spans.len() * 2);
        let mut cut = false;
        for (k, &(a, b, held)) in spans.iter().enumerate() {
            let room = max_divisions.saturating_sub(next.len() + (spans.len() - k));
            let n = if held || room == 0 {
                1
            } else {
                parts(a, b).min(room + 1)
            };
            if n == 1 {
                next.push((a, b, true));
                continue;
            }
            cut = true;
            for part in 0..n {
                #[allow(clippy::cast_precision_loss)]
                let (from, to) = (
                    if part == 0 {
                        a
                    } else {
                        a + (b - a) * part as f64 / n as f64
                    },
                    if part + 1 == n {
                        b
                    } else {
                        a + (b - a) * (part + 1) as f64 / n as f64
                    },
                );
                next.push((from, to, false));
            }
        }
        spans = next;
        if !cut {
            break;
        }
    }
    std::iter::once(lo)
        .chain(spans.into_iter().map(|(_, b, _)| b))
        .collect()
}

/// Cuts the grid lines `us` by `vs` until no cell strays past the deflection
/// from the triangles it splits into along either diagonal
/// ([`probe_strays`]). Lines sized by [`iso_breaks_over`] hold each
/// direction's chords within half the deflection, so a cell curved both ways
/// already lies within the whole of it; what is left is a twisted cell, whose
/// sides are straight and whose middle stands off its triangles. A cell that
/// strays has its two spans halved, and no other cell's.
pub(super) fn untwist_lines(
    at: &dyn Fn(f64, f64) -> Point3,
    us: &mut Vec<f64>,
    vs: &mut Vec<f64>,
    deflection: f64,
) {
    const MAX_PASSES: usize = 10;
    // A span is read again only after a cut made it: a cell between two
    // spans that both held stays as it was read.
    let (mut fresh_u, mut fresh_v) = (vec![true; us.len() - 1], vec![true; vs.len() - 1]);
    for _ in 0..MAX_PASSES {
        let (mut cut_u, mut cut_v) = (vec![false; us.len() - 1], vec![false; vs.len() - 1]);
        // Each grid point is a corner of four cells: read once.
        let corners: Vec<Vec<Point3>> = us
            .iter()
            .map(|&u| vs.iter().map(|&v| at(u, v)).collect())
            .collect();
        for i in 0..us.len() - 1 {
            for j in 0..vs.len() - 1 {
                if (cut_u[i] && cut_v[j]) || !(fresh_u[i] || fresh_v[j]) {
                    continue;
                }
                let cell = [
                    corners[i][j],
                    corners[i + 1][j],
                    corners[i][j + 1],
                    corners[i + 1][j + 1],
                ];
                let (u0, v0) = (us[i], vs[j]);
                let (du, dv) = (us[i + 1] - u0, vs[j + 1] - v0);
                let strays = PROBES.iter().any(|&(a, b)| {
                    split_stray(cell, &[at(du.madd(a, u0), dv.madd(b, v0))]) > deflection
                });
                if strays {
                    cut_u[i] = true;
                    cut_v[j] = true;
                }
            }
        }
        let more = |cut: &[bool]| cut.len() + cut.iter().filter(|&&c| c).count();
        if !cut_u.contains(&true) {
            return;
        }
        if more(&cut_u) * more(&cut_v) > GRID_MAX_CELLS {
            log::warn!(
                "twisted NURBS grid stops at {} x {} cells, its budget; it may stand off its surface by more than the requested deflection",
                us.len() - 1,
                vs.len() - 1
            );
            return;
        }
        let halve = |lines: &mut Vec<f64>, cut: &[bool]| {
            let mut next = Vec::with_capacity(lines.len() + cut.len());
            let mut fresh = Vec::with_capacity(lines.len() + cut.len());
            for (k, &line) in lines.iter().enumerate() {
                next.push(line);
                match cut.get(k) {
                    Some(&true) => {
                        next.push(0.5 * (line + lines[k + 1]));
                        fresh.extend([true, true]);
                    }
                    Some(&false) => fresh.push(false),
                    None => {}
                }
            }
            *lines = next;
            fresh
        };
        fresh_u = halve(us, &cut_u);
        fresh_v = halve(vs, &cut_v);
    }
}

/// Refine an `(n_u, n_v)` grid over `u_span` by `v_span` until each checked
/// cell strays within half the deflection of the triangles it splits into
/// along either diagonal ([`probe_strays`]). A twisted cell (a bilinear patch
/// between skew lines) has straight sides and iso-lines, which no chord test
/// sees, yet its middle stands up to a quarter of its twist,
/// |P00 + P11 - P01 - P10| / 4, off those triangles. A flat cell whose
/// parameters run unevenly across it lies in its triangles' plane, however
/// far from their diagonal, and needs nothing.
///
/// A cell's stray, from its twist or from its curvature both ways, falls
/// with the square of its size, so both directions grow by the root of what
/// the worst cell still needs, with a tenth to spare and at most double. A
/// tube's quarter whose iso-lines each hold half the deflection strays up
/// to their sum: doubling both directions for a cell a tenth over would
/// quadruple its mesh.
///
/// `chords` tells whether a count of divisions along `u` (`true`) or `v`
/// keeps the iso-line chords the starting grid was sized by. A count that
/// is no multiple of the starting one moves the grid's lines off those the
/// sizing read, and a chord can then take in two creases it held apart. Nor
/// does a multiple always hold: creases turning opposite ways can leave a
/// chord's ends alike and turn one of its parts. A step `chords` refuses
/// goes up through the multiples of the starting count to the first it
/// holds, and to the first multiple when none within the division cap does.
pub(super) fn untwisted_grid(
    at: &dyn Fn(f64, f64) -> Point3,
    ((u_lo, u_hi), (v_lo, v_hi)): ((f64, f64), (f64, f64)),
    (mut n_u, mut n_v): (usize, usize),
    deflection: f64,
    max_divisions: usize,
    chords: Option<&dyn Fn(bool, usize) -> bool>,
) -> (usize, usize) {
    let (start_u, start_v) = (n_u.max(1), n_v.max(1));
    // Every cell is read: a patch's worst cell can sit anywhere, at a
    // taper's narrow end or wherever a saddle flattens, and a spread of cells
    // can step over it.
    let half = 0.5 * deflection;
    // The worst cell's stray over the grid, and where its probes strayed
    // past half the deflection.
    let worst = |n_u: usize, n_v: usize, past: &mut Vec<(f64, (f64, f64))>| {
        #[allow(clippy::cast_precision_loss)]
        let (du, dv) = ((u_hi - u_lo) / n_u as f64, (v_hi - v_lo) / n_v as f64);
        let mut worst = 0.0_f64;
        for i in 0..n_u {
            for j in 0..n_v {
                #[allow(clippy::cast_precision_loss)]
                let corner = (du.madd(i as f64, u_lo), dv.madd(j as f64, v_lo));
                for (stray, uv) in probe_strays(at, corner, (du, dv)) {
                    worst = worst.max(stray);
                    if stray > half {
                        past.push((stray, uv));
                    }
                }
            }
        }
        worst
    };
    // How far the grid's cell holding `(u, v)` strays from its triangles there.
    let stray_at = |(n_u, n_v): (usize, usize), (u, v): (f64, f64)| {
        #[allow(clippy::cast_precision_loss)]
        let (du, dv) = ((u_hi - u_lo) / n_u as f64, (v_hi - v_lo) / n_v as f64);
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let cell = |x: f64, lo: f64, d: f64, n: usize| {
            (((x - lo) / d).floor().max(0.0) as usize).min(n - 1)
        };
        let (i, j) = (cell(u, u_lo, du, n_u), cell(v, v_lo, dv, n_v));
        #[allow(clippy::cast_precision_loss)]
        let corner =
            |a: usize, b: usize| at(du.madd((i + a) as f64, u_lo), dv.madd((j + b) as f64, v_lo));
        split_stray(
            [corner(0, 0), corner(1, 0), corner(0, 1), corner(1, 1)],
            &[at(u, v)],
        )
    };
    let holds = |along_u: bool, n: usize| chords.is_none_or(|holds| holds(along_u, n));
    // From `from` on by steps of `by`, the first count the chords hold; `from`
    // itself when none within the division cap does.
    let first_held = |from: usize, by: usize, along_u: bool| {
        let mut m = from;
        while !holds(along_u, m) {
            if m >= max_divisions {
                return from;
            }
            m = m.saturating_add(by).min(max_divisions);
        }
        m
    };
    // A whole multiple of a chain grid's count, the least past one its chords
    // hold: every multiple puts each cell's middle and quarter points on the
    // new grid's vertices or probes.
    let nested = |n: usize, along_u: bool| first_held((2 * n).min(max_divisions), n, along_u);
    // A proportional count, or the multiples of the sized count from it on,
    // whose chords each lie within one the sizing read.
    let sized = |n: usize, start: usize, along_u: bool| {
        if holds(along_u, n) {
            n
        } else {
            first_held(
                n.div_ceil(start).saturating_mul(start).min(max_divisions),
                start,
                along_u,
            )
        }
    };
    // The grids reached by whole multiples from the start, each keeping its
    // forebears' probes on its vertices or probes, and every probe of any
    // grid read that strayed past half the deflection. A proportional grid
    // between them is tried only when it holds those witnesses, and given
    // up for the chain's next multiple when it does not hold itself: a grid
    // whose lines move off a narrow peak a probe found can leave it between
    // its own probes.
    let mut chain = (n_u, n_v);
    // Each chain grid holds at least four times the cells of the one before,
    // and a grid tried between two no more than the later one, so at five
    // probes a cell the witnesses number at most some 13 times the last chain
    // grid's cells.
    // A whole multiple keeps a probe where one was, and one still straying
    // there is the same witness.
    let mut witnesses: Vec<(f64, f64)> = Vec::new();
    let mut witnessed: std::collections::HashSet<(u64, u64)> = std::collections::HashSet::new();
    let mut refined = false;
    loop {
        // The twist never takes cells past the grid's budget: a grid capped
        // down to it gives up the divisions the chord test asked for. Short of
        // a grid it could move to, it checks nothing.
        let fits_a_step = |n_u: usize, n_v: usize| {
            let next = ((n_u + 1).min(max_divisions), (n_v + 1).min(max_divisions));
            next != (n_u, n_v) && next.0 * next.1 <= GRID_MAX_CELLS
        };
        if !fits_a_step(n_u, n_v) {
            if refined {
                log::warn!(
                    "twisted NURBS grid stops at {n_u} x {n_v} cells, its budget; it may stand off its surface by more than the requested deflection"
                );
            }
            break;
        }
        let mut past = Vec::new();
        let stray = witnesses
            .iter()
            .map(|&uv| stray_at((n_u, n_v), uv))
            .fold(worst(n_u, n_v, &mut past), f64::max);
        let (held_u, held_v) = (holds(true, n_u), holds(false, n_v));
        if stray <= half && held_u && held_v {
            break;
        }
        for &(_, (u, v)) in &past {
            if witnessed.insert((u.to_bits(), v.to_bits())) {
                witnesses.push((u, v));
            }
        }
        let next = if (n_u, n_v) == chain {
            let grow = (1.1 * (stray / half).sqrt()).min(2.0);
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss
            )]
            let step = |n: usize, held: bool| {
                if stray <= half && held {
                    n
                } else {
                    (((n as f64) * grow).ceil() as usize)
                        .max(n + 1)
                        .min(max_divisions)
                }
            };
            let tried = (
                sized(step(n_u, held_u), start_u, true),
                sized(step(n_v, held_v), start_v, false),
            );
            if tried == chain || witnesses.iter().any(|&uv| stray_at(tried, uv) > half) {
                chain = (nested(n_u, true), nested(n_v, false));
                chain
            } else {
                tried
            }
        } else {
            chain = (nested(chain.0, true), nested(chain.1, false));
            chain
        };
        if next.0 * next.1 > GRID_MAX_CELLS {
            log::warn!(
                "twisted NURBS grid stops at {n_u} x {n_v} cells, its budget; it may stand off its surface by more than the requested deflection"
            );
            break;
        }
        (n_u, n_v) = next;
        refined = true;
    }
    (n_u, n_v)
}

/// How far the cell from `(u0, v0)` across `(du, dv)` strays from the two
/// triangles it splits into, along either diagonal (a mesh may take either):
/// the farthest of its middle and its quarter points. A twisted cell strays
/// most at its middle; a tapered one, whose split leaves a sliver standing
/// steep, can stray farther toward its narrow end. A flat cell lies in its
/// triangles and strays nowhere.
#[cfg(test)]
pub(super) fn cell_twist(
    at: &dyn Fn(f64, f64) -> Point3,
    (u0, v0): (f64, f64),
    (du, dv): (f64, f64),
) -> f64 {
    let corner = |a: f64, b: f64| at(du.madd(a, u0), dv.madd(b, v0));
    let corners = [
        corner(0.0, 0.0),
        corner(1.0, 0.0),
        corner(0.0, 1.0),
        corner(1.0, 1.0),
    ];
    let probes = PROBES.map(|(a, b)| corner(a, b));
    split_stray(corners, &probes)
}

/// Each of [`PROBES`] of the cell from `(u0, v0)` across `(du, dv)`, with how
/// far it strays from the cell's triangles and where it lies.
fn probe_strays(
    at: &dyn Fn(f64, f64) -> Point3,
    (u0, v0): (f64, f64),
    (du, dv): (f64, f64),
) -> [(f64, (f64, f64)); 5] {
    let uv = |a: f64, b: f64| (du.madd(a, u0), dv.madd(b, v0));
    let corner = |a: f64, b: f64| {
        let (u, v) = uv(a, b);
        at(u, v)
    };
    let corners = [
        corner(0.0, 0.0),
        corner(1.0, 0.0),
        corner(0.0, 1.0),
        corner(1.0, 1.0),
    ];
    PROBES.map(|(a, b)| (split_stray(corners, &[corner(a, b)]), uv(a, b)))
}

/// Where in a cell [`probe_strays`] reads it: its middle and quarter points.
const PROBES: [(f64, f64); 5] = [
    (0.5, 0.5),
    (0.25, 0.25),
    (0.75, 0.25),
    (0.25, 0.75),
    (0.75, 0.75),
];

/// How far the farthest of `probes` lies from the two triangles a cell with
/// corners `[p00, p10, p01, p11]` splits into, along either diagonal.
fn split_stray([p00, p10, p01, p11]: [Point3; 4], probes: &[Point3]) -> f64 {
    let split = |p: Point3, a: Point3, b: Point3, c: Point3, d: Point3| {
        point_triangle_distance(p, a, b, c).min(point_triangle_distance(p, a, c, d))
    };
    probes
        .iter()
        .map(|&p| split(p, p00, p10, p11, p01).max(split(p, p10, p11, p01, p00)))
        .fold(0.0, f64::max)
}

/// The distance from `p` to the triangle `a`, `b`, `c`; to its longest side
/// when the triangle has no area.
pub(super) fn point_triangle_distance(p: Point3, a: Point3, b: Point3, c: Point3) -> f64 {
    let segment = |a: Point3, b: Point3| {
        let ab = b - a;
        let len2 = ab.dot(ab);
        let t = if len2 > 0.0 {
            ((p - a).dot(ab) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (p - (a + ab * t)).length()
    };
    let edges = segment(a, b).min(segment(b, c)).min(segment(c, a));
    let normal = (b - a).cross(c - a);
    let area2 = normal.length();
    let scale = (b - a).length().max((c - a).length()).max((c - b).length());
    if area2 <= 1e-12 * scale * scale {
        return edges;
    }
    let n = normal * (1.0 / area2);
    let off = (p - a).dot(n);
    let q = p - n * off;
    // Inside when `q` lies on the same side of every edge as the triangle.
    let inside = [(a, b), (b, c), (c, a)]
        .iter()
        .all(|&(x, y)| (y - x).cross(q - x).dot(n) >= 0.0);
    if inside { off.abs() } else { edges }
}

/// Tessellate a NURBS surface via curvature-adaptive subdivision.
#[allow(clippy::too_many_lines)]
pub(super) fn tessellate_nurbs(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    deflection: f64,
    angular_tol: f64,
) -> TriangleMeshUV {
    let (u_lo, u_hi) = surface.domain_u();
    let (v_lo, v_hi) = surface.domain_v();

    // The grid below assumes the surface's (u, v) parameterization is
    // right-handed (dS/du x dS/dv points along `surface.normal`). A
    // left-handed parameterization (a mirrored UV domain, as the sphere-
    // corner patches build per-corner with varying contact order) would
    // otherwise mesh with the geometric normal anti-parallel to the
    // surface normal, silently inverting the face's material side in
    // volume/lighting. Detect the handedness once and mirror the
    // triangle winding when left-handed.
    // Sample the parametric handedness at a safe interior point.
    let (um, vm) = (u_lo + (u_hi - u_lo) * 0.5, v_lo + (v_hi - v_lo) * 0.5);
    let mut handedness = 1.0_f64;
    let duv = surface.first_derivatives(um, vm);
    let su = duv.1;
    let sv = duv.2;
    let nrm = safe_normal(surface, um, vm);
    if su.cross(sv).dot(nrm) < 0.0 {
        handedness = -1.0;
    }
    let mut creases = creases(surface);
    let (roots, mut cells) = loop {
        let roots = RootGrid {
            u: root_lines(u_lo, u_hi, &creases.u),
            v: root_lines(v_lo, v_hi, &creases.v),
        };
        let mut cells = Vec::with_capacity(256);
        for u_span in roots.u.windows(2) {
            for v_span in roots.v.windows(2) {
                cells.push(AdaptiveCell {
                    u_min: u_span[0],
                    u_max: u_span[1],
                    v_min: v_span[0],
                    v_max: v_span[1],
                    depth: 0,
                    children: None,
                });
            }
        }
        let n_roots = cells.len();
        for i in 0..n_roots {
            build_quadtree(surface, &mut cells, i, deflection, angular_tol, &creases);
        }
        // A tree that ran out of cells left its later roots unread, so it starts
        // again on the even grid, whose tree the budget always holds.
        if cells.len() < MAX_CELLS || (creases.u.is_empty() && creases.v.is_empty()) {
            break (roots, cells);
        }
        creases = Creases {
            u: Vec::new(),
            v: Vec::new(),
        };
    };

    conforming_pass(surface, &mut cells, &roots);

    let leaf_count = cells.iter().filter(|c| c.children.is_none()).count();
    let mut eval_cache: DetHashMap<(u64, u64), (Point3, Vec3)> = DetHashMap::default();
    let mut positions = Vec::with_capacity(leaf_count * 4);
    let mut normals = Vec::with_capacity(leaf_count * 4);
    let mut uvs: Vec<[f64; 2]> = Vec::with_capacity(leaf_count * 4);
    let mut indices = Vec::with_capacity(leaf_count * 6);
    let mut vertex_map: DetHashMap<(u64, u64), u32> = DetHashMap::default();

    let get_or_insert_vertex = |u: f64,
                                v: f64,
                                eval_cache: &mut DetHashMap<(u64, u64), (Point3, Vec3)>,
                                positions: &mut Vec<Point3>,
                                normals: &mut Vec<Vec3>,
                                uvs: &mut Vec<[f64; 2]>,
                                vertex_map: &mut DetHashMap<(u64, u64), u32>|
     -> u32 {
        let key = (u.to_bits(), v.to_bits());
        if let Some(&idx) = vertex_map.get(&key) {
            return idx;
        }
        let &mut (pos, nrm) = eval_cache.entry(key).or_insert_with(|| {
            let p = surface.evaluate(u, v);
            let n = safe_normal(surface, u, v);
            (p, n)
        });
        #[allow(clippy::cast_possible_truncation)]
        let idx = positions.len() as u32;
        positions.push(pos);
        normals.push(nrm);
        uvs.push([u, v]);
        vertex_map.insert(key, idx);
        idx
    };

    for cell in &cells {
        if cell.children.is_some() {
            continue;
        }

        let i00 = get_or_insert_vertex(
            cell.u_min,
            cell.v_min,
            &mut eval_cache,
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut vertex_map,
        );
        let i10 = get_or_insert_vertex(
            cell.u_max,
            cell.v_min,
            &mut eval_cache,
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut vertex_map,
        );
        let i11 = get_or_insert_vertex(
            cell.u_max,
            cell.v_max,
            &mut eval_cache,
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut vertex_map,
        );
        let i01 = get_or_insert_vertex(
            cell.u_min,
            cell.v_max,
            &mut eval_cache,
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut vertex_map,
        );
        if handedness > 0.0 {
            indices.push(i00);
            indices.push(i10);
            indices.push(i11);

            indices.push(i00);
            indices.push(i11);
            indices.push(i01);
        } else {
            indices.push(i00);
            indices.push(i11);
            indices.push(i10);

            indices.push(i00);
            indices.push(i01);
            indices.push(i11);
        }
    }

    TriangleMeshUV {
        mesh: TriangleMesh {
            positions,
            normals,
            indices,
        },
        uvs,
    }
}

#[cfg(test)]
mod crease_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::nurbs::surface::NurbsSurface;
    use brepkit_math::vec::Point3;

    use super::creases;

    /// A half cylinder of two quarter arcs joined at `split`, smooth across it
    /// however narrow either span.
    fn half_cylinder(split: f64) -> NurbsSurface {
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let row = |x: f64| {
            vec![
                Point3::new(x, 5.0, 0.0),
                Point3::new(x, 5.0, 5.0),
                Point3::new(x, 0.0, 5.0),
                Point3::new(x, -5.0, 5.0),
                Point3::new(x, -5.0, 0.0),
            ]
        };
        NurbsSurface::new(
            1,
            2,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, split, split, 1.0, 1.0, 1.0],
            vec![row(0.0), row(10.0)],
            vec![vec![1.0, w, 1.0, w, 1.0]; 2],
        )
        .unwrap()
    }

    #[test]
    fn a_smooth_join_in_a_narrow_span_is_no_crease() {
        for split in [0.5, 0.0001, 0.9999] {
            let found = creases(&half_cylinder(split));
            assert!(
                found.u.is_empty() && found.v.is_empty(),
                "at {split}: {:?}",
                found.v
            );
        }
    }

    /// A bumpy sheet creased along eight knots each way: its crease grid holds
    /// far more roots than the cell budget can refine.
    fn creased_bumps() -> NurbsSurface {
        let spans = 9_usize;
        let knots = || {
            let mut knots = vec![0.0; 3];
            for k in 1..spans {
                #[allow(clippy::cast_precision_loss)]
                let at = k as f64 / spans as f64;
                knots.extend([at, at]);
            }
            knots.extend([1.0; 3]);
            knots
        };
        let count = 2 * spans + 1;
        #[allow(clippy::cast_precision_loss)]
        let rows: Vec<Vec<Point3>> = (0..count)
            .map(|i| {
                (0..count)
                    .map(|j| {
                        let z = if i % 2 == 1 && j % 2 == 1 { 1.0 } else { 0.0 };
                        Point3::new(i as f64, j as f64, z)
                    })
                    .collect()
            })
            .collect();
        NurbsSurface::new(2, 2, knots(), knots(), rows, vec![vec![1.0; count]; count]).unwrap()
    }

    #[test]
    fn a_tree_out_of_cells_meshes_the_whole_face_evenly() {
        let surface = creased_bumps();
        let found = creases(&surface);
        assert_eq!((found.u.len(), found.v.len()), (8, 8), "the creases");
        let meshed = super::tessellate_nurbs(&surface, 1e-6, 0.0);
        let (mesh, uvs) = (&meshed.mesh, &meshed.uvs);
        let mut worst = 0.0_f64;
        for t in mesh.indices.chunks_exact(3) {
            let at = |i: u32| mesh.positions[i as usize];
            let uv = |i: u32| uvs[i as usize];
            let (u, v) = (
                (uv(t[0])[0] + uv(t[1])[0] + uv(t[2])[0]) / 3.0,
                (uv(t[0])[1] + uv(t[1])[1] + uv(t[2])[1]) / 3.0,
            );
            let flat = Point3::new(
                (at(t[0]).x() + at(t[1]).x() + at(t[2]).x()) / 3.0,
                (at(t[0]).y() + at(t[1]).y() + at(t[2]).y()) / 3.0,
                (at(t[0]).z() + at(t[1]).z() + at(t[2]).z()) / 3.0,
            );
            worst = worst.max((surface.evaluate(u, v) - flat).length());
        }
        // Restarting on the even grid leaves 0.015 at worst. Left to the budget,
        // the later roots stay whole and read 0.198.
        assert!(worst < 0.05, "a triangle {worst} off the surface");
    }

    #[test]
    fn two_folds_a_hair_apart_are_two_creases() {
        let (first, gap) = (4.3, 5e-6);
        let rows: Vec<Vec<Point3>> = [0.0, 10.0]
            .iter()
            .map(|&x| {
                vec![
                    Point3::new(x, 0.0, 0.0),
                    Point3::new(x, first, 0.0),
                    Point3::new(x, first + gap, gap),
                    Point3::new(x, 10.0, gap),
                ]
            })
            .collect();
        let knot = first / 10.0;
        let next = knot + (2.0_f64 * gap * gap).sqrt() / 10.0;
        let sheet = NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, knot, next, 1.0, 1.0],
            rows,
            vec![vec![1.0; 4]; 2],
        )
        .unwrap();
        assert_eq!(creases(&sheet).v.len(), 2);
    }
}

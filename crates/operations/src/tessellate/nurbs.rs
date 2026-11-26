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
    depth: u16,
    /// Indices into the cell vec; `None` means this is a leaf cell.
    children: Option<[usize; 4]>,
}

/// The depth past which a cell is divided for its distance from the
/// surface alone, not for the turn of its normals: across a crease the roots
/// do not hold, the normals disagree however small the cell gets.
const ANGULAR_DEPTH: u16 = 6;

/// The deepest a cell lies below its root. A cell there is 2^-64 of its
/// root's width, far past any feature a face's own parameters can show: a
/// cell that still strays there is refused. The deepest tree in the meshing
/// tests reaches 9, and a whole tree of 16 roots reaches the sample limit by
/// its tenth level, so a cell past it lies on a narrow branch.
const MAX_TREE_DEPTH: u16 = 64;

/// Initial grid resolution (cells per direction).
const INITIAL_CELLS: usize = 4;

/// Most crease lines the root grid takes per direction. A surface with more, such
/// as a faceted sheet, keeps the even grid there: each line multiplies the roots
/// the other way, and this many keeps them in the hundreds.
const MAX_CREASE_LINES: usize = 16;

/// The radius a cone face's columns are counted at: the radius at the end of
/// `v_range` (its distances from the apex along a generator, as
/// [`compute_v_param_range`] gives them) farther from the apex, and at least
/// 0.01. The face and solid meshers both count with it.
pub(super) fn cone_column_radius(
    cone: &brepkit_math::surfaces::ConicalSurface,
    v_range: (f64, f64),
) -> f64 {
    cone.radius_at(v_range.1.abs().max(v_range.0.abs()))
        .max(0.01)
}

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

/// Whether a NURBS curve is one straight segment: degree 1 with two
/// control points and positive weights, which runs from one point to the
/// other without turning back. Such a curve covers its chord, as a line
/// does.
pub(super) fn runs_straight(nurbs: &brepkit_math::nurbs::curve::NurbsCurve) -> bool {
    nurbs.degree() == 1
        && nurbs.control_points().len() == 2
        && nurbs.weights().iter().all(|&w| w > 0.0)
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
            EdgeCurve::NurbsCurve(nurbs) if runs_straight(nurbs) => 1,
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

    // Each edge's start as the wire runs it.
    let mut wire_pts = Vec::new();
    if let Ok(wire) = topo.wire(face_data.outer_wire()) {
        for oe in wire.edges() {
            if let Ok(edge) = topo.edge(oe.edge())
                && let Ok(vertex) = topo.vertex(oe.oriented_start(edge))
            {
                wire_pts.push(vertex.point());
            }
        }
    }

    if wire_pts.len() < 3 {
        return (-FRAC_PI_2, FRAC_PI_2);
    }

    // A face of latitudes and meridians (the grid's faces) spans exactly
    // the latitudes its boundary reaches: a band between two circles, a cap
    // whose seam runs to its pole.
    let mut lo = f64::MAX;
    let mut hi = f64::MIN;
    if let Ok(wire) = topo.wire(face_data.outer_wire()) {
        for oe in wire.edges() {
            let Ok(edge) = topo.edge(oe.edge()) else {
                continue;
            };
            let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) else {
                continue;
            };
            let (p0, p1) = (sv.point(), ev.point());
            let (t0, t1) = edge.curve().domain_with_endpoints(p0, p1);
            for k in 0..=8 {
                let t = (t1 - t0).madd(f64::from(k) / 8.0, t0);
                let v = sphere
                    .project_point(edge.curve().evaluate_with_endpoints(t, p0, p1))
                    .1;
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
    }
    if hi - lo > 1e-9 {
        return (lo.max(-FRAC_PI_2), hi.min(FRAC_PI_2));
    }

    let avg_v: f64 = wire_pts
        .iter()
        .map(|pt| sphere.project_point(*pt).1)
        .sum::<f64>()
        / wire_pts.len() as f64;

    // A cap under one latitude: the wire winds counter-clockwise about the
    // sphere's own axis round a cap toward its north pole. Read about that
    // axis, not +Z, which a cap turned on its side sees edge on.
    let centre = sphere.center();
    let axis = sphere.z_axis();
    let signed_area: f64 = (0..wire_pts.len())
        .map(|i| {
            let a = wire_pts[i] - centre;
            let b = wire_pts[(i + 1) % wire_pts.len()] - centre;
            a.cross(b).dot(axis)
        })
        .sum();
    if signed_area > 0.0 {
        (avg_v, FRAC_PI_2)
    } else {
        (-FRAC_PI_2, avg_v)
    }
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

    // Each side's middle against its chord: where the parameters crowd
    // along a straight side, its middle lies on the chord all the same.
    let edge_mids = [
        surface.evaluate(u_mid, v_min),
        surface.evaluate(u_mid, v_max),
        surface.evaluate(u_min, v_mid),
        surface.evaluate(u_max, v_mid),
    ];
    let chords = [(p00, p10), (p01, p11), (p00, p01), (p10, p11)];
    let edge_sags = edge_mids
        .into_iter()
        .zip(chords)
        .map(|(mid, (a, b))| point_segment_distance(mid, a, b));

    let diag = (p11 - p00).length().max((p10 - p01).length());
    let normal_sag = max_normal_dev * diag * 0.5;

    // The middle and quarter points against the triangles the cell splits
    // into: the distance from the mesh, not from the corners' average, which
    // misses a twist and counts the crowding of the parameters as sag.
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
    let read = [p00, p10, p11, p01]
        .iter()
        .chain(&edge_mids)
        .chain(&probes)
        .all(|p| p.x().is_finite() && p.y().is_finite() && p.z().is_finite());
    let twist = if read {
        split_stray([p00, p10, p01, p11], &probes)
    } else {
        f64::NAN
    };

    // A point that is not a number would drop out of a plain maximum, and
    // the cell would pass.
    edge_sags.chain([normal_sag, twist]).fold(0.0_f64, |m, e| {
        if m.is_nan() || e.is_nan() {
            f64::NAN
        } else {
            m.max(e)
        }
    })
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

/// Build the adaptive quadtree below the cell `root`, until each leaf lies
/// within `threshold` of the surface. The cells are read from a stack of
/// their own, each branch whole before the next, as a recursion would read
/// them, so that a deep branch takes no call stack.
///
/// # Errors
///
/// Returns [`crate::OperationsError::TooManySamples`] if the tree would take
/// more cells than the sample limit, a cell past [`MAX_TREE_DEPTH`] or one
/// that cannot be halved still strays, or the surface is not finite in a
/// cell.
#[allow(clippy::similar_names)]
fn build_quadtree(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    cells: &mut Vec<AdaptiveCell>,
    root: usize,
    threshold: f64,
    angular_tol: f64,
    creases: &Creases,
) -> Result<(), crate::OperationsError> {
    let mut work = vec![root];
    while let Some(cell_idx) = work.pop() {
        let cell = &cells[cell_idx];
        let u_min = cell.u_min;
        let u_max = cell.u_max;
        let v_min = cell.v_min;
        let v_max = cell.v_max;
        let depth = cell.depth;

        let error = cell_refinement_error(surface, u_min, u_max, v_min, v_max, creases);
        if !error.is_finite() {
            return Err(super::endless_samples(
                "a B-spline face's cell, whose surface is not finite",
                threshold,
            ));
        }
        let angular_exceeded = depth < ANGULAR_DEPTH
            && cell_exceeds_angular(surface, u_min, u_max, v_min, v_max, angular_tol, creases);
        if error <= threshold && !angular_exceeded {
            continue;
        }

        let u_mid = 0.5 * (u_min + u_max);
        let v_mid = 0.5 * (v_min + v_max);
        check_cells(
            cells.len(),
            depth,
            (u_min, u_mid, u_max),
            (v_min, v_mid, v_max),
            threshold,
        )?;
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
        work.extend([c0 + 3, c0 + 2, c0 + 1, c0]);
    }
    Ok(())
}

/// Refuses four more cells for a tree of `count`, from the halving of a cell
/// at `depth` at the middles `u` and `v`, past the sample limit, at
/// [`MAX_TREE_DEPTH`], or where a middle does not part its cell: such a cell
/// cannot be halved.
fn check_cells(
    count: usize,
    depth: u16,
    (u_min, u_mid, u_max): (f64, f64, f64),
    (v_min, v_mid, v_max): (f64, f64, f64),
    deflection: f64,
) -> Result<(), crate::OperationsError> {
    if depth >= MAX_TREE_DEPTH
        || !(u_min < u_mid && u_mid < u_max && v_min < v_mid && v_mid < v_max)
    {
        return Err(super::endless_samples("a B-spline face's cell", deflection));
    }
    super::check_total("a B-spline face's cells", count as u64 + 4, deflection)
}

/// Conforming pass: ensure no more than 1 level difference between adjacent
/// leaf cells. Each pass halves the leaves beside one two levels deeper, so
/// the deepest level is reached in as many passes as the tree is deep.
///
/// # Errors
///
/// Returns [`crate::OperationsError::TooManySamples`] if the tree would take
/// more cells than the sample limit.
fn conforming_pass(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    cells: &mut Vec<AdaptiveCell>,
    roots: &RootGrid,
    deflection: f64,
) -> Result<(), crate::OperationsError> {
    loop {
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
            return Ok(());
        }

        for &cell_idx in &to_subdivide {
            if cells[cell_idx].children.is_some() {
                continue;
            }
            force_subdivide(surface, cells, cell_idx, deflection)?;
        }
    }
}

/// Check if a leaf cell needs conforming subdivision (neighbor is 2+ levels deeper).
#[allow(clippy::similar_names)]
fn needs_conforming_subdivision(
    cells: &[AdaptiveCell],
    roots: &RootGrid,
    depth: u16,
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
fn find_leaf_depth_at(cells: &[AdaptiveCell], roots: &RootGrid, u: f64, v: f64) -> Option<u16> {
    let root = roots.holding(u, v)?;
    find_leaf_depth_below(cells, root, u, v)
}

/// The depth of the leaf holding a point below the cell `idx`, read down
/// the tree; one past a cell none of whose children holds it.
fn find_leaf_depth_below(cells: &[AdaptiveCell], idx: usize, u: f64, v: f64) -> Option<u16> {
    let holds = |cell: &AdaptiveCell| {
        !(u < cell.u_min || u > cell.u_max || v < cell.v_min || v > cell.v_max)
    };
    let mut cell = &cells[idx];
    if !holds(cell) {
        return None;
    }
    loop {
        match cell.children {
            None => return Some(cell.depth),
            Some(children) => match children.iter().find(|&&child| holds(&cells[child])) {
                Some(&child) => cell = &cells[child],
                None => return Some(cell.depth + 1),
            },
        }
    }
}

/// Force-subdivide a leaf cell (for conforming pass, no curvature check).
#[allow(clippy::similar_names)]
fn force_subdivide(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    cells: &mut Vec<AdaptiveCell>,
    cell_idx: usize,
    deflection: f64,
) -> Result<(), crate::OperationsError> {
    let cell = &cells[cell_idx];
    let u_min = cell.u_min;
    let u_max = cell.u_max;
    let v_min = cell.v_min;
    let v_max = cell.v_max;
    let depth = cell.depth;

    let u_mid = 0.5 * (u_min + u_max);
    let v_mid = 0.5 * (v_min + v_max);
    check_cells(
        cells.len(),
        depth,
        (u_min, u_mid, u_max),
        (v_min, v_mid, v_max),
        deflection,
    )?;
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

    let _ = surface;
    Ok(())
}

/// Mesh a NURBS surface that closes on itself in both directions (a torus
/// image) as a structured grid over its whole domain. The adaptive quadtree
/// keeps one-level steps between neighbouring cells, which leave T-junction
/// cracks, and cannot see that the domain's opposite edges meet. Here every
/// row is sampled alike, and the far seam rows copy the near rows' positions,
/// so a caller that welds by position closes both seams exactly.
///
/// # Errors
///
/// Returns [`crate::OperationsError::TooManySamples`] if the grid the
/// deflection needs is past the sample limit.
pub(super) fn tessellate_periodic_nurbs_grid(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    deflection: f64,
    angular_tol: f64,
) -> Result<TriangleMeshUV, crate::OperationsError> {
    let (u_lo, u_hi) = surface.domain_u();
    let (v_lo, v_hi) = surface.domain_v();
    // Fewer than three divisions of a closed direction collapse its seam
    // copies onto the cells' other side.
    let n_u = iso_divisions(surface, true, deflection, angular_tol)?.max(3);
    let n_v = iso_divisions(surface, false, deflection, angular_tol)?.max(3);
    super::check_grid(n_u + 1, n_v + 1, deflection)?;
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
    Ok(TriangleMeshUV {
        mesh: TriangleMesh {
            positions,
            normals,
            indices,
        },
        uvs,
    })
}

/// Divisions of one parameter direction that keep every iso-line chord of a
/// few sampled rows within half the deflection (a grid cell's diagonal sags
/// further than its sides) and within the angular tolerance.
fn iso_divisions(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    along_u: bool,
    deflection: f64,
    angular_tol: f64,
) -> Result<usize, crate::OperationsError> {
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
        (4096, super::MAX_DIVISIONS),
    )
}

/// [`iso_divisions`] over a parameter box: `ranges` is the span divided and
/// the span its sampled rows sit across, and `at` / `normal_at` take
/// `(u, v)` (a caller on a periodic surface wraps them). A straight direction
/// (a ruling) needs one division. The normals' turn is weighed across every
/// chord but one that ends on a pole.
///
/// The normals' turn is met up to `turn_cap` divisions: a turn that does not
/// settle there, as across a crease, is left, and the chords' distance from
/// the surface alone takes the count on from there.
///
/// # Errors
///
/// Returns [`crate::OperationsError::TooManySamples`] if the chords' distance
/// holds at no power of two up to `max_divisions`.
#[allow(clippy::too_many_arguments)]
pub(super) fn iso_divisions_over(
    at: &dyn Fn(f64, f64) -> Point3,
    normal_at: &dyn Fn(f64, f64) -> Vec3,
    along_u: bool,
    ranges: ((f64, f64), (f64, f64)),
    deflection: f64,
    angular_tol: f64,
    (turn_cap, max_divisions): (usize, usize),
) -> Result<usize, crate::OperationsError> {
    let mut n = 1;
    while n < turn_cap
        && !iso_chords_hold(at, normal_at, along_u, ranges, deflection, angular_tol, n)
    {
        n *= 2;
    }
    while !iso_chords_hold(at, normal_at, along_u, ranges, deflection, 0.0, n) {
        n = n.saturating_mul(2);
        if n > max_divisions {
            return Err(too_many_divisions(
                "a B-spline face's divisions",
                n,
                max_divisions,
                deflection,
            ));
        }
    }
    Ok(n)
}

/// The refusal of `what`, which would take `divisions` divisions of one
/// direction, past `max_divisions`.
#[allow(clippy::cast_precision_loss)]
fn too_many_divisions(
    what: &str,
    divisions: usize,
    max_divisions: usize,
    deflection: f64,
) -> crate::OperationsError {
    super::too_many_samples(
        what,
        divisions as f64 + 1.0,
        max_divisions as f64 + 1.0,
        deflection,
    )
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
                // Each point is read against the chord, not against the
                // chord's own point at its fraction: a span whose parameter
                // runs unevenly along a straight chord lies on it. An
                // inflected span can cross its chord at its middle and stand
                // off it either side, so its quarter points count too.
                let sag = [0.25, 0.5, 0.75]
                    .iter()
                    .map(|&f| off_chord(point(t0 + (t1 - t0) * f, o), p0, p1))
                    .fold(0.0, f64::max);
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
///
/// # Errors
///
/// Returns [`crate::OperationsError::TooManySamples`] if the spans the
/// deflection needs are more than `max_divisions`.
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
) -> Result<Vec<f64>, crate::OperationsError> {
    const ROWS: usize = 8;
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
        let (mut sag, mut turn, mut reach) = (0.0_f64, 0.0_f64, 0.0_f64);
        for row in 0..ROWS {
            #[allow(clippy::cast_precision_loss)]
            let o = o_lo + (o_hi - o_lo) * (row as f64 + 0.5) / ROWS as f64;
            let (pa, pb) = (point(a, o), point(b, o));
            reach = reach.max((pb - pa).length());
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
        // A span shorter than a tenth of the deflection on every row is not
        // cut for its turn, as a spline edge's is not: where the surface
        // barely moves, lines a hair apart cannot be told apart in (u, v).
        let turn_ratio = if angular_tol > 0.0 && reach > 0.1 * deflection {
            turn / angular_tol
        } else {
            0.0
        };
        parts_needed(sag / (0.5 * deflection), turn_ratio)
    };

    // Level by level: each level reads again only the spans the one before
    // cut. A span that never holds is cut on every level, so the count
    // passes `max_divisions` and is refused, as it passes it: the spans
    // still to come take one each at least.
    let mut spans = vec![(lo, hi, false)];
    loop {
        let mut next = Vec::with_capacity((spans.len() * 2).min(max_divisions + 1));
        let mut cut = false;
        for (k, &(a, b, held)) in spans.iter().enumerate() {
            let n = if held { 1 } else { parts(a, b) };
            if n == 1 {
                next.push((a, b, true));
                continue;
            }
            // A span with no parameter strictly between its ends cannot be
            // cut: its parts would be the span again and empty spans.
            let mid = 0.5 * (a + b);
            if !(a.min(b) < mid && mid < a.max(b)) {
                return Err(super::endless_samples(
                    "a B-spline face's grid lines",
                    deflection,
                ));
            }
            let projected = next.len() + n + (spans.len() - k - 1);
            if projected > max_divisions {
                return Err(too_many_divisions(
                    "a B-spline face's grid lines",
                    projected,
                    max_divisions,
                    deflection,
                ));
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
    Ok(std::iter::once(lo)
        .chain(spans.into_iter().map(|(_, b, _)| b))
        .collect())
}

/// Cuts the grid lines `us` by `vs` until no cell strays past the deflection
/// from the triangles it splits into along either diagonal
/// ([`probe_strays`]). Lines sized by [`iso_breaks_over`] hold each
/// direction's chords within half the deflection, so a cell curved both ways
/// already lies within the whole of it; what is left is a twisted cell, whose
/// sides are straight and whose middle stands off its triangles. A cell that
/// strays has its two spans halved, and no other cell's. A cell `bound`
/// gives a stray for is read by it, and any other at its [`PROBES`].
///
/// # Errors
///
/// Returns [`crate::OperationsError::TooManySamples`] if the grid the
/// halving needs is past the sample limit, or the surface is not finite at
/// a line's crossing or a probe. A stray falls with the square of a cell's
/// size, so the passes end where the surface is smooth.
pub(super) fn untwist_lines(
    at: &dyn Fn(f64, f64) -> Point3,
    us: &mut Vec<f64>,
    vs: &mut Vec<f64>,
    deflection: f64,
    bound: Option<CellBound>,
) -> Result<(), crate::OperationsError> {
    // A span is read again only after a cut made it: a cell between two
    // spans that both held stays as it was read.
    let (mut fresh_u, mut fresh_v) = (vec![true; us.len() - 1], vec![true; vs.len() - 1]);
    loop {
        let (mut cut_u, mut cut_v) = (vec![false; us.len() - 1], vec![false; vs.len() - 1]);
        // Each grid point is a corner of four cells: read once.
        let corners: Vec<Vec<Point3>> = us
            .iter()
            .map(|&u| vs.iter().map(|&v| at(u, v)).collect())
            .collect();
        if !corners.iter().flatten().all(|&p| finite(p)) {
            return Err(not_finite(deflection));
        }
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
                let read = bound.and_then(|bound| bound((u0, us[i + 1]), (v0, vs[j + 1])));
                let strays = if let Some(stray) = read {
                    stray.is_nan() || stray > deflection
                } else {
                    let mut strays = false;
                    for &(a, b) in &PROBES {
                        let stray = split_stray(cell, &[at(du.madd(a, u0), dv.madd(b, v0))]);
                        if stray.is_nan() {
                            return Err(not_finite(deflection));
                        }
                        if stray > deflection {
                            strays = true;
                            break;
                        }
                    }
                    strays
                };
                if strays {
                    cut_u[i] = true;
                    cut_v[j] = true;
                }
            }
        }
        let more = |cut: &[bool]| cut.len() + cut.iter().filter(|&&c| c).count();
        if !cut_u.contains(&true) {
            return Ok(());
        }
        super::check_grid(more(&cut_u) + 1, more(&cut_v) + 1, deflection)?;
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

/// [`untwisted_grid_keeping`] with room for [`MAX_WITNESSES`] witnesses and
/// no bound, each cell read at its probes.
///
/// # Errors
///
/// As [`untwisted_grid_keeping`].
#[cfg(test)]
pub(super) fn untwisted_grid(
    at: &dyn Fn(f64, f64) -> Point3,
    spans: ((f64, f64), (f64, f64)),
    start: (usize, usize),
    deflection: f64,
    max_divisions: usize,
    chords: Option<&dyn Fn(bool, usize) -> bool>,
) -> Result<(usize, usize), crate::OperationsError> {
    let mut held = 0;
    untwisted_grid_keeping(
        at,
        spans,
        start,
        deflection,
        max_divisions,
        (chords, None),
        (MAX_WITNESSES, &mut held),
    )
}

/// [`untwisted_grid_keeping`] with room for [`MAX_WITNESSES`] witnesses,
/// reading each cell `bound` gives a stray for by it, and any other at its
/// [`PROBES`].
///
/// # Errors
///
/// As [`untwisted_grid_keeping`].
pub(super) fn untwisted_grid_bounded(
    at: &dyn Fn(f64, f64) -> Point3,
    bound: CellBound,
    spans: ((f64, f64), (f64, f64)),
    start: (usize, usize),
    deflection: f64,
    max_divisions: usize,
) -> Result<(usize, usize), crate::OperationsError> {
    let mut held = 0;
    untwisted_grid_keeping(
        at,
        spans,
        start,
        deflection,
        max_divisions,
        (None, Some(bound)),
        (MAX_WITNESSES, &mut held),
    )
}

/// The most probes the grid search ([`untwisted_grid_keeping`]) keeps as
/// witnesses. A pass whose
/// strays pass the room left keeps the farthest of them, and the search
/// then takes only whole multiples of its grids, which need no witnesses:
/// each keeps every probe of the one before on its vertices or probes.
const MAX_WITNESSES: usize = 1 << 20;

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
/// holds.
///
/// A cell `bound` gives a stray for is read by it, and any other at its
/// [`PROBES`]. The search keeps at most `room` probes as witnesses, and
/// sets `held` to the most probes it held at once: while it scans a grid,
/// up to twice the room it has left, and one more.
///
/// # Errors
///
/// Returns [`crate::OperationsError::TooManySamples`] if a direction needs
/// more than `max_divisions` divisions, the grid more cells than the sample
/// limit, or the surface is not finite at a point a cell is read at.
#[allow(clippy::too_many_lines)]
pub(super) fn untwisted_grid_keeping(
    at: &dyn Fn(f64, f64) -> Point3,
    ((u_lo, u_hi), (v_lo, v_hi)): ((f64, f64), (f64, f64)),
    (mut n_u, mut n_v): (usize, usize),
    deflection: f64,
    max_divisions: usize,
    (chords, bound): (Option<ChordTest>, Option<CellBound>),
    (room, held): (usize, &mut usize),
) -> Result<(usize, usize), crate::OperationsError> {
    let (start_u, start_v) = (n_u.max(1), n_v.max(1));
    // Every cell is read: a patch's worst cell can sit anywhere, at a
    // taper's narrow end or wherever a saddle flattens, and a spread of cells
    // can step over it.
    let half = 0.5 * deflection;
    // The farthest `keep` of `past`, by how far each strayed.
    let farthest = |past: &mut Vec<(f64, (f64, f64))>, keep: usize| {
        if past.len() > keep {
            if keep > 0 {
                past.select_nth_unstable_by(keep - 1, |x, y| y.0.total_cmp(&x.0));
            }
            past.truncate(keep);
        }
    };
    // The worst cell's stray over the grid, with the farthest `keep` of the
    // probes that strayed past half the deflection, whether more did, and
    // the most it held at once.
    let worst = |n_u: usize, n_v: usize, past: &mut Vec<(f64, (f64, f64))>, keep: usize| {
        #[allow(clippy::cast_precision_loss)]
        let (du, dv) = ((u_hi - u_lo) / n_u as f64, (v_hi - v_lo) / n_v as f64);
        let mut worst = 0.0_f64;
        let mut more = false;
        let mut most = 0;
        for i in 0..n_u {
            for j in 0..n_v {
                #[allow(clippy::cast_precision_loss)]
                let corner = (du.madd(i as f64, u_lo), dv.madd(j as f64, v_lo));
                // A bound reads the whole cell, so it leaves no witness.
                let cell = ((corner.0, corner.0 + du), (corner.1, corner.1 + dv));
                if let Some(stray) = bound.and_then(|bound| bound(cell.0, cell.1)) {
                    worst = worst.max(stray);
                    continue;
                }
                for (stray, uv) in probe_strays(at, corner, (du, dv)) {
                    if stray.is_nan() {
                        return (stray, more, most);
                    }
                    worst = worst.max(stray);
                    if stray > half {
                        past.push((stray, uv));
                        most = most.max(past.len());
                        if past.len() > keep.saturating_mul(2) {
                            farthest(past, keep);
                            more = true;
                        }
                    }
                }
            }
        }
        if past.len() > keep {
            farthest(past, keep);
            more = true;
        }
        (worst, more, most)
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
    let refused = |divisions: usize| {
        too_many_divisions(
            "a twisted B-spline face's grid",
            divisions,
            max_divisions,
            deflection,
        )
    };
    // From `from` on by steps of `by`, the first count the chords hold;
    // refused when none up to `max_divisions` does.
    let first_held = |from: usize, by: usize, along_u: bool| {
        let mut m = from;
        while !holds(along_u, m) {
            m = m.saturating_add(by);
            if m > max_divisions {
                return Err(refused(m));
            }
        }
        Ok(m)
    };
    // A whole multiple of a chain grid's count, the least past one its chords
    // hold: every multiple puts each cell's middle and quarter points on the
    // new grid's vertices or probes.
    let nested = |n: usize, along_u: bool| first_held(n.saturating_mul(2), n, along_u);
    // A proportional count, or the multiples of the sized count from it on,
    // whose chords each lie within one the sizing read.
    let sized = |n: usize, start: usize, along_u: bool| {
        if holds(along_u, n) {
            Ok(n)
        } else {
            first_held(n.div_ceil(start).saturating_mul(start), start, along_u)
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
    // A whole multiple keeps a probe where one was, and one still straying
    // there is the same witness.
    let mut witnesses: Vec<(f64, f64)> = Vec::new();
    let mut witnessed: std::collections::HashSet<(u64, u64)> = std::collections::HashSet::new();
    // Whether the witnesses hold every probe that strayed: a grid between
    // two whole multiples is tried only while they do.
    let mut whole = true;
    // A grid past the division or sample limit.
    let past_limits = |(a, b): (usize, usize)| -> Result<(), crate::OperationsError> {
        if a.max(b) > max_divisions {
            return Err(refused(a.max(b)));
        }
        super::check_grid(a + 1, b + 1, deflection)
    };
    loop {
        let mut past = Vec::new();
        let keep = if whole { room - witnesses.len() } else { 0 };
        let (scanned, more, most) = worst(n_u, n_v, &mut past, keep);
        *held = (*held).max(witnesses.len() + most);
        let stray = witnesses
            .iter()
            .map(|&uv| stray_at((n_u, n_v), uv))
            .fold(scanned, |a, b| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.max(b)
                }
            });
        if stray.is_nan() {
            return Err(not_finite(deflection));
        }
        let (held_u, held_v) = (holds(true, n_u), holds(false, n_v));
        if stray <= half && held_u && held_v {
            break;
        }
        // The next grid is the chain's next multiple, or one tried between;
        // when both are past the limits, the search is refused before it
        // keeps this pass's witnesses. A pass that kept only the farthest of
        // its strays goes on from a multiple of its own grid, which keeps
        // every probe it read on its vertices or probes, the strays it let go
        // among them, though it is a grid tried between.
        let doubled = if (n_u, n_v) == chain || more {
            nested(n_u, true).and_then(|a| Ok((a, nested(n_v, false)?)))
        } else {
            nested(chain.0, true).and_then(|a| Ok((a, nested(chain.1, false)?)))
        };
        let tried = if whole && !more && (n_u, n_v) == chain {
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
                    (((n as f64) * grow).ceil() as usize).max(n + 1)
                }
            };
            Some((
                sized(step(n_u, held_u), start_u, true)?,
                sized(step(n_v, held_v), start_v, false)?,
            ))
        } else {
            None
        };
        let fits = |grid: &Result<(usize, usize), crate::OperationsError>| {
            grid.as_ref().is_ok_and(|&g| past_limits(g).is_ok())
        };
        if !fits(&doubled) {
            match tried {
                Some(t) => past_limits(t)?,
                None => return doubled.and_then(|d| past_limits(d).map(|()| d)),
            }
        }
        for &(_, (u, v)) in &past {
            if witnessed.insert((u.to_bits(), v.to_bits())) {
                witnesses.push((u, v));
            }
        }
        whole &= !more;
        let next = match tried {
            Some(t)
                if t != chain
                    && past_limits(t).is_ok()
                    && witnesses.iter().all(|&uv| stray_at(t, uv) <= half) =>
            {
                t
            }
            _ => {
                chain = doubled?;
                chain
            }
        };
        past_limits(next)?;
        (n_u, n_v) = next;
    }
    Ok((n_u, n_v))
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

/// A bound on how far the cell from `(u0, u1)` by `(v0, v1)` strays from
/// the two triangles it splits into along either diagonal, where the patch
/// gives one: a cell with none is read at its [`PROBES`].
pub(super) type CellBound<'a> = &'a dyn Fn((f64, f64), (f64, f64)) -> Option<f64>;

/// Whether a count of divisions along `u` (`true`) or `v` keeps the chords
/// a grid search's starting grid was sized by ([`untwisted_grid_keeping`]).
pub(super) type ChordTest<'a> = &'a dyn Fn(bool, usize) -> bool;

/// One direction of a degree-1 B-spline patch's parameter box, read at the
/// pace of its weights: [`Pace::at`] gives the parameter at a fraction of
/// the box at that pace.
///
/// One span whose weights are one direction's factor times the other's,
/// `w_ij = a_i b_j`, is the patch of equal weights on its control points
/// with each direction's fraction `t` of the span moved to `a1 t / (a0 (1 -
/// t) + a1 t)`. Its pace undoes that move: an even grid in it is an even
/// grid of the patch of equal weights, whose cells halve their twist as
/// they halve, however the weights crowd. Other weights are read at the
/// pace of the geometric means of each direction's two edges.
///
/// The move reads only the ratio `a1 / a0`, so a scale common to all the
/// weights leaves the pace as it is. A ratio, a span or a box whose moved
/// fractions are not finite and rising gives the even pace.
#[derive(Clone, Copy, Debug)]
pub(super) struct Pace {
    knots: (f64, f64),
    /// The far end's weight over the near end's.
    ratio: f64,
    from: f64,
    to: f64,
}

impl Pace {
    /// The pace over `lo` to `hi` of a span from `knots.0` to `knots.1`
    /// whose far end weighs `ratio` times its near end; `None` where its
    /// moved fractions are not finite and rising.
    fn new(knots: (f64, f64), ratio: f64, (lo, hi): (f64, f64)) -> Option<Self> {
        let mut pace = Self {
            knots,
            ratio,
            from: 0.0,
            to: 1.0,
        };
        (pace.from, pace.to) = (pace.moved(lo), pace.moved(hi));
        let rising = pace.from.is_finite() && pace.to.is_finite() && pace.from < pace.to;
        (ratio.is_finite() && ratio > 0.0 && rising).then_some(pace)
    }

    /// An even pace over `lo` to `hi`.
    pub(super) fn even(lo: f64, hi: f64) -> Self {
        Self {
            knots: (lo, hi),
            ratio: 1.0,
            from: 0.0,
            to: 1.0,
        }
    }

    /// The span's moved fraction at `u`.
    fn moved(&self, u: f64) -> f64 {
        let ((k0, k1), r) = (self.knots, self.ratio);
        let t = (u - k0) / (k1 - k0);
        r * t / 1.0.madd(1.0 - t, r * t)
    }

    /// The parameter at the fraction `s` of the box at this pace.
    pub(super) fn at(&self, s: f64) -> f64 {
        let ((k0, k1), r) = (self.knots, self.ratio);
        let g = (self.to - self.from).madd(s, self.from);
        (k1 - k0).madd(g / r.madd(1.0 - g, g), k0)
    }

    /// The fraction of the box at this pace at the parameter `u`.
    pub(super) fn fraction(&self, u: f64) -> f64 {
        (self.moved(u) - self.from) / (self.to - self.from)
    }
}

/// The paces of a degree-1 patch of one span over the box `u` by `v`
/// ([`Pace`]), and even paces for any other patch.
pub(super) fn weight_paces(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    u: (f64, f64),
    v: (f64, f64),
) -> (Pace, Pace) {
    let even = (Pace::even(u.0, u.1), Pace::even(v.0, v.1));
    let (ku, kv, w) = (surface.knots_u(), surface.knots_v(), surface.weights());
    let one_span = surface.degree_u() == 1
        && surface.degree_v() == 1
        && ku.len() == 4
        && kv.len() == 4
        && w.len() == 2
        && w.iter()
            .all(|row| row.len() == 2 && row.iter().all(|&x| x > 0.0 && x.is_finite()));
    if !one_span {
        return even;
    }
    // The ratio of the geometric means `√(f0 f1)` and `√(n0 n1)`, from the
    // roots of the weights' ratios, so no product of weights overflows or
    // underflows.
    let ratio = |(n0, n1): (f64, f64), (f0, f1): (f64, f64)| (f0 / n0).sqrt() * (f1 / n1).sqrt();
    (
        Pace::new(
            (ku[1], ku[2]),
            ratio((w[0][0], w[0][1]), (w[1][0], w[1][1])),
            u,
        )
        .unwrap_or(even.0),
        Pace::new(
            (kv[1], kv[2]),
            ratio((w[0][0], w[1][0]), (w[0][1], w[1][1])),
            v,
        )
        .unwrap_or(even.1),
    )
}

/// [`untwist_lines`] for a B-spline face, read by its cells' twist bounds
/// ([`bilinear_cell_stray`]) at its weights' pace ([`weight_paces`]): the
/// lines it adds halve each span at its pace, and the lines it was given
/// keep their parameters. Where two lines read one fraction of the pace, or
/// lines it adds read back to one parameter, the lines are cut in the
/// face's own parameters instead, still read by the cells' bounds.
///
/// # Errors
///
/// As [`untwist_lines`].
pub(super) fn untwist_paced_lines(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    at: &dyn Fn(f64, f64) -> Point3,
    us: &mut Vec<f64>,
    vs: &mut Vec<f64>,
    deflection: f64,
) -> Result<(), crate::OperationsError> {
    let (Some(&u_lo), Some(&u_hi), Some(&v_lo), Some(&v_hi)) =
        (us.first(), us.last(), vs.first(), vs.last())
    else {
        return Ok(());
    };
    // A box with no width has no fractions to read a pace at.
    if !(u_lo < u_hi && v_lo < v_hi) {
        return untwist_lines(at, us, vs, deflection, None);
    }
    let (pace_u, pace_v) = weight_paces(surface, (u_lo, u_hi), (v_lo, v_hi));
    // A given line keeps its own parameter, not one read back from the pace.
    let given = |lines: &[f64], pace: &Pace| -> DetHashMap<u64, f64> {
        lines
            .iter()
            .map(|&x| (pace.fraction(x).to_bits(), x))
            .collect()
    };
    let (given_u, given_v) = (given(us, &pace_u), given(vs, &pace_v));
    let to_u = |s: f64| {
        given_u
            .get(&s.to_bits())
            .copied()
            .unwrap_or_else(|| pace_u.at(s))
    };
    let to_v = |t: f64| {
        given_v
            .get(&t.to_bits())
            .copied()
            .unwrap_or_else(|| pace_v.at(t))
    };
    let paced = |s: f64, t: f64| at(to_u(s), to_v(t));
    let bound = |(s0, s1): (f64, f64), (t0, t1): (f64, f64)| {
        bilinear_cell_stray(surface, (to_u(s0), to_u(s1)), (to_v(t0), to_v(t1)))
    };
    let rising = |lines: &[f64]| lines.windows(2).all(|pair| pair[0] < pair[1]);
    let unpaced = |us: &mut Vec<f64>, vs: &mut Vec<f64>| {
        let bound = |u: (f64, f64), v: (f64, f64)| bilinear_cell_stray(surface, u, v);
        untwist_lines(at, us, vs, deflection, Some(&bound))
    };
    let mut ss: Vec<f64> = us.iter().map(|&u| pace_u.fraction(u)).collect();
    let mut ts: Vec<f64> = vs.iter().map(|&v| pace_v.fraction(v)).collect();
    if !(rising(&ss) && rising(&ts)) {
        return unpaced(us, vs);
    }
    untwist_lines(&paced, &mut ss, &mut ts, deflection, Some(&bound))?;
    let paced_us: Vec<f64> = ss.into_iter().map(to_u).collect();
    let paced_vs: Vec<f64> = ts.into_iter().map(to_v).collect();
    if !(rising(&paced_us) && rising(&paced_vs)) {
        return unpaced(us, vs);
    }
    (*us, *vs) = (paced_us, paced_vs);
    Ok(())
}

/// A bound on how far a cell of a degree-1 B-spline patch strays from the
/// two triangles it splits into along either diagonal. `None` for another
/// patch, for a cell across a knot, and where a weight is not positive.
///
/// In one span the cell is a rational bilinear patch on its corners `P`,
/// with weights `h`, the surface's denominators there: each of its points
/// is `Σ λ P`, with `λ_ij = h_ij b_i b_j / Σ h b b`. Such a point stands off
/// the triangles of the diagonal `P00 P11` by at most `min(λ10, λ01) |T|`,
/// with `T = P00 + P11 - P10 - P01`, and off those of `P10 P01` by at most
/// `min(λ00, λ11) |T|`. Each pair's product is at most a quarter of `h10 h01
/// / (√(h00 h11) + √(h10 h01))²`, so the cell strays at most `|T| max(1, ρ)
/// / (2 (1 + ρ))`, with `ρ = √(h00 h11 / (h10 h01))`. That is a quarter of
/// the twist where each weight is one direction's factor times the other's,
/// however their pace crowds, and less than half of it for any weights.
///
/// A twist in the plane of `P00`, `P10` and `P01` moves no point off it, so
/// a second bound reads only the twist's part `γ` along that plane's
/// normal. Each point lies in the tetrahedron on the four corners. Where
/// the corners seen along the normal make a convex quad, the tetrahedron
/// lies between the two splits, so a point stands off either at most the
/// splits' gap, which is widest where the diagonals cross: `|γ|` times the
/// fraction of `P00 P11` there. A flat cell strays nowhere.
///
/// The bound holds the rounding of what it is read from. The weights are
/// scaled by a power of two and read only by their ratios, so a scale
/// common to all of them leaves it as it is. The corners are read from the
/// span's net less its first point, so the rounding follows the cell, not
/// its distance from the origin: the twist carries the rounding of the
/// corners' differences, and the bound the rounding of the corners, within
/// `16 ε` of the net's farthest point from its first each. A corner that is
/// not finite, or a weight that is not, gives no bound.
pub(super) fn bilinear_cell_stray(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    (u0, u1): (f64, f64),
    (v0, v1): (f64, f64),
) -> Option<f64> {
    if surface.degree_u() != 1 || surface.degree_v() != 1 {
        return None;
    }
    // The span of `knots` that holds `lo` to `hi`, and its ends.
    let span = |knots: &[f64], lo: f64, hi: f64| {
        let s = knots.partition_point(|&k| k <= lo).checked_sub(1)?;
        let (a, b) = (*knots.get(s)?, *knots.get(s + 1)?);
        (a <= lo && hi <= b && a < b).then_some((s, a, b))
    };
    let (su, ua, ub) = span(surface.knots_u(), u0, u1)?;
    let (sv, va, vb) = span(surface.knots_v(), v0, v1)?;
    let (weights, points) = (surface.weights(), surface.control_points());
    let block = [
        [
            *weights.get(su - 1)?.get(sv - 1)?,
            *weights.get(su - 1)?.get(sv)?,
        ],
        [*weights.get(su)?.get(sv - 1)?, *weights.get(su)?.get(sv)?],
    ];
    if !block.iter().flatten().all(|&w| w > 0.0 && w.is_finite()) {
        return None;
    }
    let net = [
        *points.get(su - 1)?.get(sv - 1)?,
        *points.get(su - 1)?.get(sv)?,
        *points.get(su)?.get(sv - 1)?,
        *points.get(su)?.get(sv)?,
    ];
    // The net read from its first point, so that rounding follows the
    // cell's size, not its distance from the origin.
    let local = net.map(|p| p - net[0]);
    let most = local.iter().map(|d| d.length()).fold(0.0, f64::max);
    let unit =
        brepkit_math::vec::unit_scale(block.iter().flatten().fold(0.0, |m: f64, &w| m.max(w)));
    let block = block.map(|row| row.map(|w| w * unit));
    // The corner at `(u, v)` less the net's first point, with the surface's
    // denominator there.
    let corner = |u: f64, v: f64| {
        let (a, b) = ((u - ua) / (ub - ua), (v - va) / (vb - va));
        let terms = [
            (1.0 - a) * (1.0 - b) * block[0][0],
            (1.0 - a) * b * block[0][1],
            a * (1.0 - b) * block[1][0],
            a * b * block[1][1],
        ];
        let h = terms.iter().sum::<f64>();
        let sum = terms
            .iter()
            .zip(&local)
            .fold(Vec3::new(0.0, 0.0, 0.0), |sum, (&c, &d)| sum + d * c);
        (Point3::new(sum.x() / h, sum.y() / h, sum.z() / h), h)
    };
    let [(p00, h00), (p10, h10), (p01, h01), (p11, h11)] = [
        corner(u0, v0),
        corner(u1, v0),
        corner(u0, v1),
        corner(u1, v1),
    ];
    if ![p00, p10, p01, p11].iter().all(|p| finite(*p)) {
        return None;
    }
    let eps = f64::EPSILON;
    let (d1, d2) = (p00 - p10, p11 - p01);
    let twist = d1 + d2;
    // Each difference and their sum round within half `ε` of the parts.
    let doubt = 2.0 * eps * (d1.length() + d2.length());
    // `ρ` and `1 / ρ` from the roots of the denominators' ratios. A weight
    // too small to keep its digits beside the largest, or a ratio past the
    // range of a double, leaves the bound for any weights: half the twist.
    let rho = (h00 / h10).sqrt() * (h11 / h01).sqrt();
    let q = rho.min(rho.recip());
    let kept = block.iter().flatten().all(|&w| w >= f64::MIN_POSITIVE);
    let q = if kept && q >= 0.0 && q.is_finite() {
        q
    } else {
        0.0
    };
    let by_weights = (twist.length() + doubt) / (2.0 * (1.0 + q)) * eps.madd(16.0, 1.0);
    let bound = splits_gap(p10 - p00, p01 - p00, twist, doubt)
        .map_or(by_weights, |gap| gap.min(by_weights));
    let bound = (16.0 * eps).madd(most, bound);
    bound.is_finite().then_some(bound)
}

/// Whether each coordinate of `p` is finite.
fn finite(p: Point3) -> bool {
    p.x().is_finite() && p.y().is_finite() && p.z().is_finite()
}

/// The refusal of a grid whose surface is not finite at a point it reads.
fn not_finite(deflection: f64) -> crate::OperationsError {
    super::endless_samples("a face's grid, whose surface is not finite", deflection)
}

/// The widest gap between the two splits of a cell from a corner along the
/// sides `e1` and `e2`, with twist `twist`, measured along the normal of
/// the sides' plane, where the corners seen along that normal make a convex
/// quad: `None` otherwise, and where the rounding of the sides' plane could
/// tip it by more than a twentieth of a radian. `doubt` bounds how far
/// `twist` rounds from the twist of the corners.
///
/// The twist's part in the plane is `α e1 + β e2`, with `α = (T × e2) · n /
/// |e1 × e2|` and `β = (e1 × T) · n / |e1 × e2|`. These hold none of the
/// cancellation of the sides' Gram determinant `|e1|² |e2|² - (e1 · e2)²`,
/// which rounds to zero on a sliver whose sides are long beside its width.
/// Each term carries the rounding of what it is read from, and the gap is
/// read at the ends of those doubts that widen it.
fn splits_gap(e1: Vec3, e2: Vec3, twist: Vec3, doubt: f64) -> Option<f64> {
    let eps = f64::EPSILON;
    let (l1, l2, lt) = (e1.length(), e2.length(), twist.length());
    let across = e1.cross(e2);
    let area2 = across.length();
    // Each term of the cross product rounds within `ε` of the two products
    // it is the difference of, and the sides round within half `ε` each.
    let off = 4.0 * eps * l1 * l2;
    let clear = area2 > 64.0 * off;
    if !clear {
        return None;
    }
    let normal = across * (1.0 / area2);
    // How far the normal can tip from that of the corners' sides.
    let tip = (3.0 * off).madd(1.0 / area2, 2.0 * eps);
    let lift = twist.dot(normal).abs() + tip.madd(lt + doubt, (eps * lt).madd(4.0, doubt));
    let rel = (2.0 * off).madd(1.0 / area2, 4.0 * eps);
    // The lower end of `α` or `β` from its triple product along the normal.
    let low = |side: f64, triple: f64| {
        let part = triple / area2;
        let spread = 1.25 * side * eps.madd(8.0, tip).madd(lt, doubt) / area2;
        part - rel.madd(part.abs(), spread)
    };
    let alpha = low(l2, twist.cross(e2).dot(normal));
    let beta = low(l1, e1.cross(twist).dot(normal));
    // The diagonals cross inside both when the quad is convex, `P00 P11` at
    // the fraction `1 / (2 + α + β)`.
    let convex = alpha > -1.0 && beta > -1.0 && alpha + beta > -1.0;
    convex.then(|| lift / (2.0 + alpha + beta) * eps.madd(4.0, 1.0))
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
/// corners `[p00, p10, p01, p11]` splits into, along either diagonal: not a
/// number where a corner or a probe is not finite, which a plain maximum
/// would drop.
fn split_stray([p00, p10, p01, p11]: [Point3; 4], probes: &[Point3]) -> f64 {
    if ![p00, p10, p01, p11]
        .iter()
        .chain(probes)
        .all(|&p| finite(p))
    {
        return f64::NAN;
    }
    let split = |p: Point3, a: Point3, b: Point3, c: Point3, d: Point3| {
        point_triangle_distance(p, a, b, c).min(point_triangle_distance(p, a, c, d))
    };
    probes
        .iter()
        .map(|&p| split(p, p00, p10, p11, p01).max(split(p, p10, p11, p01, p00)))
        .fold(0.0, f64::max)
}

/// The distance from `p` to the segment from `a` to `b`.
fn point_segment_distance(p: Point3, a: Point3, b: Point3) -> f64 {
    let ab = b - a;
    let len2 = ab.dot(ab);
    let t = if len2 > 0.0 {
        ((p - a).dot(ab) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (p - (a + ab * t)).length()
}

/// The distance from `p` to the triangle `a`, `b`, `c`, with the doubt the
/// rounding of its plane leaves, and never more than to its nearest side.
///
/// The normal is the cross product of the triangle's two shorter sides:
/// any two sides give the same normal, but each of its terms rounds to
/// within a few units in the last place of the two products it is the
/// difference of, and the shorter sides give the smaller products. Where
/// those products are still large beside the normal, as on a sliver whose
/// longest side is near the sum of the others, the normal can tip, and the
/// distance from the plane takes that tip in. The distance is read from the
/// corner nearest `p`, so the tip moves it least.
pub(super) fn point_triangle_distance(p: Point3, a: Point3, b: Point3, c: Point3) -> f64 {
    let edges = point_segment_distance(p, a, b)
        .min(point_segment_distance(p, b, c))
        .min(point_segment_distance(p, c, a));
    let sides = [b - a, c - b, a - c];
    let longest = (0..3)
        .max_by(|&i, &j| sides[i].dot(sides[i]).total_cmp(&sides[j].dot(sides[j])))
        .unwrap_or(0);
    // Consecutive sides round the triangle give its normal, `(b - a) x (c - a)`.
    let (s, t) = (sides[(longest + 1) % 3], sides[(longest + 2) % 3]);
    let normal = s.cross(t);
    let area2 = normal.length();
    if !(area2 > 0.0 && area2.is_finite()) {
        return edges;
    }
    let n = normal * (1.0 / area2);
    let terms = [
        (s.y() * t.z()).abs() + (s.z() * t.y()).abs(),
        (s.z() * t.x()).abs() + (s.x() * t.z()).abs(),
        (s.x() * t.y()).abs() + (s.y() * t.x()).abs(),
    ];
    let tip = terms
        .into_iter()
        .zip([n.x(), n.y(), n.z()])
        .map(|(term, along)| term * (1.0 - along * along).max(0.0).sqrt())
        .sum::<f64>()
        * (4.0 * f64::EPSILON)
        / area2;
    let d = [p - a, p - b, p - c]
        .into_iter()
        .min_by(|x, y| x.dot(*x).total_cmp(&y.dot(*y)))
        .unwrap_or(p - a);
    let off = d.dot(n);
    let rounding = 4.0
        * f64::EPSILON
        * ((d.x() * n.x()).abs() + (d.y() * n.y()).abs() + (d.z() * n.z()).abs());
    let doubt = d.length().madd(tip, rounding);
    let q = p - n * off;
    // Inside when `q` lies on the same side of every edge as the triangle.
    let inside = [(a, b), (b, c), (c, a)]
        .iter()
        .all(|&(x, y)| (y - x).cross(q - x).dot(n) >= 0.0);
    if inside {
        (off.abs() + doubt).min(edges)
    } else {
        edges
    }
}

/// The parameters of a point of a face's boundary on the face's surface.
///
/// The point lies on the surface to the tolerance of the boundary's edges,
/// which a file's or a fitted section's edges can pass (a section 8e-5 off a
/// fitted wing skin). Where the proof of a nearest point does not settle
/// for such a point, the nearest point found, unproven, still places it:
/// without it the point, and the face's mesh around it, is lost.
pub(super) fn project_boundary_point(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    point: Point3,
) -> Result<brepkit_math::nurbs::projection::SurfaceProjection, brepkit_math::MathError> {
    use brepkit_math::nurbs::projection::{
        project_point_to_surface, project_point_to_surface_from,
    };
    project_point_to_surface(surface, point, 1e-6).or_else(|err| {
        // An unbounded tolerance takes the search's point without the
        // proof; Newton from there settles it.
        let size = {
            let b = surface.aabb();
            (b.max - b.min).length()
        };
        project_point_to_surface(surface, point, f64::MAX)
            .and_then(|q| project_point_to_surface_from(surface, point, (q.u, q.v), 1e-6))
            .ok()
            .filter(|q| q.distance <= 1e-3 * size.max(1.0))
            .ok_or(err)
    })
}

/// Tessellate a NURBS surface via curvature-adaptive subdivision.
///
/// # Errors
///
/// Returns [`crate::OperationsError::TooManySamples`] if the tree the
/// deflection needs is past the sample limit.
#[allow(clippy::too_many_lines)]
pub(super) fn tessellate_nurbs(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    deflection: f64,
    angular_tol: f64,
) -> Result<TriangleMeshUV, crate::OperationsError> {
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
    let creases = creases(surface);
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
        build_quadtree(surface, &mut cells, i, deflection, angular_tol, &creases)?;
    }

    conforming_pass(surface, &mut cells, &roots, deflection)?;

    let corners = leaf_corners(
        &cells,
        super::MAX_FACE_SAMPLES,
        super::MAX_LINE_SAMPLES,
        deflection,
    )?;
    drop(cells);
    let positions = corners
        .uvs
        .iter()
        .map(|&[u, v]| surface.evaluate(u, v))
        .collect();
    let normals = corners
        .uvs
        .iter()
        .map(|&[u, v]| safe_normal(surface, u, v))
        .collect();
    let mut indices = Vec::with_capacity(corners.quads.len() * 6);
    for [i00, i10, i11, i01] in corners.quads {
        if handedness > 0.0 {
            indices.extend([i00, i10, i11, i00, i11, i01]);
        } else {
            indices.extend([i00, i11, i10, i00, i01, i11]);
        }
    }

    Ok(TriangleMeshUV {
        mesh: TriangleMesh {
            positions,
            normals,
            indices,
        },
        uvs: corners.uvs,
    })
}

/// The corners of a tree's leaves, each once, in the order the mesh numbers
/// them, and the corners of each leaf as indices into them.
struct LeafCorners {
    uvs: Vec<[f64; 2]>,
    quads: Vec<[u32; 4]>,
}

/// The corners of the leaves of `cells`, refused before the mesh takes room
/// for them when the corners or the leaves pass `limit`, or the corners on
/// one parameter line pass `line_limit`.
///
/// The cell limit does not hold the corners: a leaf halved beside one that
/// is not puts a corner at the middle of their shared side, so the corners
/// can outnumber the cells. Each leaf gives two triangles, so the leaf count
/// holds the triangles to twice the limit, as a grid's are. A grid's row is
/// held to the edge limit, and so is each line of the tree.
fn leaf_corners(
    cells: &[AdaptiveCell],
    limit: usize,
    line_limit: usize,
    deflection: f64,
) -> Result<LeafCorners, crate::OperationsError> {
    let refused = |what: &str, count: usize, limit: usize| {
        super::too_many_samples(what, count as f64, limit as f64, deflection)
    };
    let mut on_u_lines: DetHashMap<u64, usize> = DetHashMap::default();
    let mut on_v_lines: DetHashMap<u64, usize> = DetHashMap::default();
    let mut numbers: DetHashMap<(u64, u64), u32> = DetHashMap::default();
    let mut uvs: Vec<[f64; 2]> = Vec::new();
    let mut quads: Vec<[u32; 4]> = Vec::new();
    for cell in cells.iter().filter(|c| c.children.is_none()) {
        if quads.len() == limit {
            return Err(refused(
                "a B-spline face's mesh cells",
                limit.saturating_add(1),
                limit,
            ));
        }
        let mut quad = [0_u32; 4];
        for (number, (u, v)) in quad.iter_mut().zip([
            (cell.u_min, cell.v_min),
            (cell.u_max, cell.v_min),
            (cell.u_max, cell.v_max),
            (cell.u_min, cell.v_max),
        ]) {
            let next = uvs.len();
            *number = match numbers.entry((u.to_bits(), v.to_bits())) {
                std::collections::hash_map::Entry::Occupied(found) => *found.get(),
                std::collections::hash_map::Entry::Vacant(slot) => {
                    let past = || refused("a B-spline face's mesh vertices", next + 1, limit);
                    if next == limit {
                        return Err(past());
                    }
                    let on_u = on_u_lines.entry(u.to_bits()).or_insert(0);
                    *on_u += 1;
                    let on_v = on_v_lines.entry(v.to_bits()).or_insert(0);
                    *on_v += 1;
                    let most = (*on_u).max(*on_v);
                    if most > line_limit {
                        return Err(refused(
                            "a B-spline face's mesh along one parameter line",
                            most,
                            line_limit,
                        ));
                    }
                    uvs.push([u, v]);
                    *slot.insert(u32::try_from(next).map_err(|_| past())?)
                }
            };
        }
        quads.push(quad);
    }
    Ok(LeafCorners { uvs, quads })
}

#[cfg(test)]
mod crease_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::nurbs::surface::NurbsSurface;
    use brepkit_math::vec::Point3;

    use super::{
        AdaptiveCell, MAX_TREE_DEPTH, check_cells, creases, force_subdivide, leaf_corners,
        point_triangle_distance,
    };

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

    /// A bumpy sheet creased along eight knots each way.
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

    /// The creased sheet's tree takes the crease grid's 100 roots and refines
    /// each leaf to the deflection. At 4e-4 it takes about 290,000 leaves, and
    /// each triangle lies within the deflection. A tree held to 200,000 cells
    /// and meshed again on the even grid lies 38 times the deflection off.
    #[test]
    fn a_creased_sheet_meshes_within_its_deflection() {
        let surface = creased_bumps();
        let found = creases(&surface);
        assert_eq!((found.u.len(), found.v.len()), (8, 8), "the creases");
        let deflection = 4e-4;
        let meshed = super::tessellate_nurbs(&surface, deflection, 0.0).unwrap();
        let (mesh, uvs) = (&meshed.mesh, &meshed.uvs);
        let leaves = mesh.indices.len() / 6;
        assert!(leaves > 200_000, "{leaves} leaves");
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
        assert!(worst <= deflection, "a triangle {worst} off the surface");
    }

    /// A patch 1e100 across each way, linear in its parameters but for a
    /// spike of height 1 at one corner, over a span 1e-100 wide each way:
    /// the triangles from the spike's top stand off the patch by most of its
    /// height until the cells at the corner are near the deflection across,
    /// some 340 halvings down.
    fn corner_spike() -> NurbsSurface {
        let (t, d) = (1e-100, 1e100);
        let knots = vec![0.0, 0.0, 0.0, t, 2.0 * t, d, d, d];
        // Each control point over its Greville abscissa keeps the patch
        // linear in its parameters.
        let greville = [0.0, 0.5 * t, 1.5 * t, 0.5 * (2.0 * t + d), d];
        let points = greville
            .iter()
            .enumerate()
            .map(|(i, &gu)| {
                greville
                    .iter()
                    .enumerate()
                    .map(|(j, &gv)| {
                        let z = if i == 0 && j == 0 { 1.0 } else { 0.0 };
                        Point3::new(gu, gv, z)
                    })
                    .collect()
            })
            .collect();
        NurbsSurface::new(2, 2, knots.clone(), knots, points, vec![vec![1.0; 5]; 5]).unwrap()
    }

    /// The spike drives the tree down one corner. On a 256 KB stack the
    /// tree is refused at its depth bound, with no overflow.
    #[test]
    fn a_spike_at_a_corner_is_refused_at_the_depth_bound_on_a_small_stack() {
        let found = std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(|| super::tessellate_nurbs(&corner_spike(), 1e-3, 0.0).map(|m| m.uvs.len()))
            .unwrap()
            .join()
            .unwrap();
        assert!(
            matches!(
                &found,
                Err(crate::OperationsError::TooManySamples { what, .. })
                    if what == "a B-spline face's cell"
            ),
            "{found:?}"
        );
    }

    /// A patch with a control point that is not a number is refused at its
    /// first cell, not divided down to the sample limit.
    #[test]
    fn a_surface_that_is_not_finite_is_refused_at_once() {
        let points = vec![
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
            vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, f64::NAN)],
        ];
        let surface = NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            points,
            vec![vec![1.0; 2]; 2],
        )
        .unwrap();
        let started = std::time::Instant::now();
        let found = super::tessellate_nurbs(&surface, 0.1, 0.0).map(|m| m.uvs.len());
        assert!(
            matches!(found, Err(crate::OperationsError::TooManySamples { .. })),
            "{found:?}"
        );
        assert!(started.elapsed().as_secs() < 1, "{:?}", started.elapsed());
    }

    /// A flat square patch with a weight of 1e-80 at one corner crowds its
    /// parameters there, but its sides and iso-lines stay straight and its
    /// cells flat, so the roots hold it.
    #[test]
    fn a_corner_crowded_by_its_weight_meshes_on_its_roots() {
        let points = vec![
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
            vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
        ];
        let surface = NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            points,
            vec![vec![1e-80, 1.0], vec![1.0, 1.0]],
        )
        .unwrap();
        let mesh = super::tessellate_nurbs(&surface, 0.1, 0.0).unwrap();
        assert_eq!(mesh.mesh.indices.len() / 6, 16, "the leaves");
    }

    /// A tree past the sample limit is refused before it takes the cells,
    /// and so is a cell at the depth bound, or one whose middle does not
    /// part it.
    #[test]
    fn a_tree_past_the_sample_limit_is_refused() {
        let halves = (0.0, 0.5, 1.0);
        let refused = |found: Result<(), crate::OperationsError>| {
            matches!(found, Err(crate::OperationsError::TooManySamples { .. }))
        };
        assert!(check_cells(9_999_996, 0, halves, halves, 1e-3).is_ok());
        assert!(refused(check_cells(9_999_997, 0, halves, halves, 1e-3)));
        assert!(check_cells(4, MAX_TREE_DEPTH - 1, halves, halves, 1e-3).is_ok());
        assert!(refused(check_cells(
            4,
            MAX_TREE_DEPTH,
            halves,
            halves,
            1e-3
        )));
        let tiny = (1.0, 1.0, 1.0 + f64::EPSILON);
        assert!(refused(check_cells(4, 0, tiny, halves, 1e-3)));
    }

    /// A root halved twice, then each other quarter of a quarter halved once
    /// more: 53 cells and 40 leaves, but 65 corners, since each leaf halved
    /// beside whole ones puts corners at the middles of its sides.
    #[test]
    fn corners_past_the_limit_are_refused_though_the_cells_are_within_it() {
        let surface = creased_bumps();
        let root = AdaptiveCell {
            u_min: 0.0,
            u_max: 1.0,
            v_min: 0.0,
            v_max: 1.0,
            depth: 0,
            children: None,
        };
        let mut cells = vec![AdaptiveCell {
            children: None,
            ..root
        }];
        let mut next = 0;
        while next < cells.len() {
            if cells[next].depth < 2 {
                force_subdivide(&surface, &mut cells, next, 1e-3).unwrap();
            }
            next += 1;
        }
        let quarters: Vec<usize> = (0..cells.len())
            .filter(|&i| cells[i].children.is_none())
            .collect();
        for i in quarters {
            let (u, v) = (cells[i].u_min * 4.0, cells[i].v_min * 4.0);
            if (u + v) % 2.0 < 0.5 {
                force_subdivide(&surface, &mut cells, i, 1e-3).unwrap();
            }
        }
        assert_eq!(cells.len(), 53, "the cells");
        let corners = leaf_corners(&cells, 65, 65, 1e-3).unwrap();
        assert_eq!((corners.uvs.len(), corners.quads.len()), (65, 40));
        let refused = |found: Result<_, crate::OperationsError>| {
            matches!(found, Err(crate::OperationsError::TooManySamples { .. }))
        };
        assert!(refused(leaf_corners(&cells, 64, 64, 1e-3)));
        let alike: Vec<AdaptiveCell> = (0..5)
            .map(|_| AdaptiveCell {
                children: None,
                ..root
            })
            .collect();
        assert!(leaf_corners(&alike, 5, 5, 1e-3).is_ok());
        assert!(
            refused(leaf_corners(&alike, 4, 4, 1e-3)),
            "five leaves past four"
        );
    }

    /// A flat bilinear patch over `corners`, `[[p00, p01], [p10, p11]]`.
    fn flat_patch(corners: [[Point3; 2]; 2], weights: [[f64; 2]; 2]) -> NurbsSurface {
        NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            corners.iter().map(|row| row.to_vec()).collect(),
            weights.iter().map(|row| row.to_vec()).collect(),
        )
        .unwrap()
    }

    /// A flat strip 1e9 long and 1e-4 wide: its cells' triangles are slivers
    /// 1e13 times as long as wide, but their plane is exact, so the roots
    /// hold the strip at a deflection of 1e-8.
    #[test]
    fn a_flat_sliver_meshes_on_its_roots() {
        let p = |x: f64, y: f64| Point3::new(x, y, 0.0);
        let strip = flat_patch(
            [[p(0.0, 0.0), p(0.0, 1e-4)], [p(1e9, 0.0), p(1e9, 1e-4)]],
            [[1.0; 2]; 2],
        );
        let mesh = super::tessellate_nurbs(&strip, 1e-8, 0.0).unwrap();
        assert_eq!(mesh.mesh.indices.len() / 6, 16, "the leaves");
        let inside = p(2.5e8, 2.5e-5);
        assert!(point_triangle_distance(inside, p(0.0, 0.0), p(1e9, 0.0), p(1e9, 1e-4)) < 1e-12);
    }

    /// A flat strip 2^20 long and 2^-14 wide along a slant, S(u, v) =
    /// (2^20 u, 2^20 u + 2^-14 v, 2^-14 v): its cells' plane is exact, and
    /// its roots hold it at a deflection of 1e-9. A point 2^-30 (1, -1, 1)
    /// off the strip's half reads at least its distance, 1.6e-9, and nowhere
    /// near the half's sides.
    #[test]
    fn a_slanted_flat_sliver_meshes_on_its_roots() {
        let (long, wide) = (2.0_f64.powi(20), 2.0_f64.powi(-14));
        let (a, b, c, d) = (
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(long, long, 0.0),
            Point3::new(0.0, wide, wide),
            Point3::new(long, long + wide, wide),
        );
        let strip = flat_patch([[a, c], [b, d]], [[1.0; 2]; 2]);
        let mesh = super::tessellate_nurbs(&strip, 1e-9, 0.0).unwrap();
        assert_eq!(mesh.mesh.indices.len() / 6, 16, "the leaves");
        let step = 2.0_f64.powi(-30);
        let off = Point3::new(
            0.5 * long + step,
            0.5 * long + 0.25 * wide - step,
            0.25 * wide + step,
        );
        let found = point_triangle_distance(off, a, b, d);
        let lies = step * 3.0_f64.sqrt();
        assert!(found >= 0.999 * lies && found < 1e-8, "{found}");
    }

    /// A tilted sliver's plane is in doubt, so a point off it reads at least
    /// as far as it lies, and a triangle with no area reads by its sides.
    #[test]
    fn a_tilted_sliver_reads_no_nearer_than_it_lies() {
        let axis = brepkit_math::vec::Vec3::new(1.0, 2.0, 3.0) * (1.0 / 14.0_f64.sqrt());
        let (sin, cos) = 0.7_f64.sin_cos();
        let turn = |x: f64, y: f64, z: f64| {
            let v = brepkit_math::vec::Vec3::new(x, y, z);
            let r = v * cos + axis.cross(v) * sin + axis * (axis.dot(v) * (1.0 - cos));
            Point3::new(r.x(), r.y(), r.z())
        };
        for (long, wide) in [(1.0, 1e-6), (1e6, 1e-6), (1e9, 1e-4)] {
            let (a, b, c) = (
                turn(0.0, 0.0, 0.0),
                turn(long, 0.0, 0.0),
                turn(long, wide, 0.0),
            );
            let off = 1e-3 * wide;
            let found = point_triangle_distance(turn(0.75 * long, 0.5 * wide, off), a, b, c);
            assert!(
                found >= 0.999 * off && found <= wide,
                "{long} by {wide}: {found}"
            );
        }
        let line = point_triangle_distance(
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
        );
        assert!((line - 1.0).abs() < 1e-15, "{line}");
    }

    /// Flat patches whose parameters crowd toward one side: a square whose
    /// weights at `u = 0` are 2^-20, and a tilted cubic sheet whose control
    /// points bunch there. Their sides are straight and their cells flat, so
    /// a cell's middles lie on its chords and triangles however its
    /// parameters crowd, and the roots hold them.
    #[test]
    fn a_flat_patch_with_crowded_parameters_meshes_on_its_roots() {
        let p = |x: f64, y: f64| Point3::new(x, y, 0.0);
        let tiny = 2.0_f64.powi(-20);
        let square = flat_patch(
            [[p(0.0, 0.0), p(0.0, 1.0)], [p(1.0, 0.0), p(1.0, 1.0)]],
            [[tiny, tiny], [1.0, 1.0]],
        );
        let mesh = super::tessellate_nurbs(&square, 0.1, 0.0).unwrap();
        assert_eq!(mesh.mesh.indices.len() / 6, 16, "the square's leaves");
        let tilt = |x: f64, y: f64| Point3::new(0.6 * x, y, 0.8 * x);
        let rows = [0.0, 0.001, 0.002, 1.0]
            .iter()
            .map(|&x| vec![tilt(x, 0.0), tilt(x, 1.0)])
            .collect();
        let sheet = NurbsSurface::new(
            3,
            1,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            rows,
            vec![vec![1.0; 2]; 4],
        )
        .unwrap();
        let mesh = super::tessellate_nurbs(&sheet, 1e-6, 0.0).unwrap();
        assert_eq!(mesh.mesh.indices.len() / 6, 16, "the sheet's leaves");
    }

    /// A root halved four times down its side at `u = 0`: 17 corners along
    /// that line, and 68 in all. The line passes a limit of 16 though the
    /// mesh does not pass its own.
    #[test]
    fn corners_past_the_line_limit_are_refused() {
        let surface = creased_bumps();
        let mut cells = vec![AdaptiveCell {
            u_min: 0.0,
            u_max: 1.0,
            v_min: 0.0,
            v_max: 1.0,
            depth: 0,
            children: None,
        }];
        let mut next = 0;
        while next < cells.len() {
            if cells[next].depth < 4 && cells[next].u_min <= 0.0 {
                force_subdivide(&surface, &mut cells, next, 1e-3).unwrap();
            }
            next += 1;
        }
        let corners = leaf_corners(&cells, 1000, 17, 1e-3).unwrap();
        let on_side = corners.uvs.iter().filter(|uv| uv[0] <= 0.0).count();
        assert_eq!((on_side, corners.uvs.len()), (17, 68));
        assert!(matches!(
            leaf_corners(&cells, 1000, 16, 1e-3),
            Err(crate::OperationsError::TooManySamples { .. })
        ));
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

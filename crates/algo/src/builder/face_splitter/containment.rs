//! Point containment tests for UV-space hole detection.

use brepkit_math::curves2d::Curve2D;
use brepkit_math::region2d::{Boundary2, point_in_region};
use brepkit_math::vec::{Point2, Vec2, Vec3};
use brepkit_topology::edge::EdgeCurve;

use super::super::split_types::OrientedPCurveEdge;
use super::sampling::sample_wire_loop_uv;

/// Build a UV polygon for a hole wire from its edges' endpoint UVs (chords),
/// not from `sample_wire_loop_uv`. The pcurve of a projected arc edge can be a
/// `Nurbs2D` whose parameter domain traces a path inconsistent with its stored
/// `start_uv`/`end_uv` (the pcurve was fitted in a different frame), so sampling
/// it yields a self-crossing garbage polygon. The stored endpoint UVs ARE
/// correct and chain edge-to-edge, so the chord polygon faithfully approximates
/// the hole for a point-in-polygon containment test (a rounded-rect's corner
/// arcs are clipped to chords, well away from any interior test point).
fn hole_chord_polygon(hole: &[OrientedPCurveEdge]) -> Vec<Point2> {
    let mut pts: Vec<Point2> = Vec::with_capacity(hole.len() + 1);
    for e in hole {
        pts.push(e.start_uv);
        // Densify a curved edge with its chord midpoint so a large corner arc
        // keeps a closer-to-true hull than a single chord. The chord midpoint
        // lies on the concave side of the arc, so it stays inside the true arc,
        // never outside — the conservative direction for a hole-containment
        // polygon (a point judged inside this under-approximation is genuinely
        // inside the hole).
        if !matches!(e.pcurve, Curve2D::Line(_)) {
            pts.push(Point2::new(
                0.5 * (e.start_uv.x() + e.end_uv.x()),
                0.5 * (e.start_uv.y() + e.end_uv.y()),
            ));
        }
    }
    pts
}

/// Check if a UV point is inside any of the inner wire (hole) polygons.
pub(super) fn is_inside_any_hole(pt: &Point2, inner_wires: &[Vec<OrientedPCurveEdge>]) -> bool {
    for hole in inner_wires {
        // Prefer the chord polygon (endpoint-derived). Fall back to the sampled
        // polygon only when the hole degenerates to < 3 endpoints.
        let chord = hole_chord_polygon(hole);
        if chord.len() >= 3 {
            if super::super::classify_2d::point_in_polygon_2d(*pt, &chord) {
                return true;
            }
            continue;
        }
        let hole_pts = sample_wire_loop_uv(hole);
        if hole_pts.len() >= 3 && super::super::classify_2d::point_in_polygon_2d(*pt, &hole_pts) {
            return true;
        }
    }
    false
}

/// How near a hole's boundary a seed may lie and still be read outside it.
const SEED_CLEARANCE: f64 = 1e-7;

/// A hole's boundary flattened into the plane frame, each line a segment and
/// each circle or ellipse the arc it is. `None` when an edge is a NURBS curve.
pub(super) fn exact_hole(
    hole: &[OrientedPCurveEdge],
    frame: &super::super::plane_frame::PlaneFrame,
) -> Option<Vec<Boundary2>> {
    let along = |d: Vec3| Vec2::new(d.dot(frame.u_axis()), d.dot(frame.v_axis()));
    hole.iter()
        .map(|e| {
            // The edge's native orientation: `domain_with_endpoints` takes
            // the positive span from the first point to the second.
            let (s3, e3) = if e.forward {
                (e.start_3d, e.end_3d)
            } else {
                (e.end_3d, e.start_3d)
            };
            match &e.curve_3d {
                EdgeCurve::Line => Some(Boundary2::Segment(frame.project(s3), frame.project(e3))),
                EdgeCurve::Circle(c) => {
                    let (t0, t1) = e.curve_3d.domain_with_endpoints(s3, e3);
                    Some(Boundary2::Arc {
                        center: frame.project(c.center()),
                        u: along(c.u_axis()),
                        v: along(c.v_axis()),
                        a: c.radius(),
                        b: c.radius(),
                        t0,
                        t1,
                    })
                }
                EdgeCurve::Ellipse(el) => {
                    let (t0, t1) = e.curve_3d.domain_with_endpoints(s3, e3);
                    Some(Boundary2::Arc {
                        center: frame.project(el.center()),
                        u: along(el.u_axis()),
                        v: along(el.v_axis()),
                        a: el.semi_major(),
                        b: el.semi_minor(),
                        t0,
                        t1,
                    })
                }
                EdgeCurve::NurbsCurve(_) => None,
            }
        })
        .collect()
}

/// Find a UV point inside the outer wire but outside all holes.
///
/// Steps inward from each outer-wire edge midpoint, then from each vertex,
/// toward the outer polygon's centroid in small increments, returning the
/// first candidate inside the outer wire and outside every hole. Falls back
/// to an outer-wire vertex midpoint, then the raw centroid.
pub(super) fn find_point_outside_holes(
    outer_pts: &[Point2],
    inner_wires: &[Vec<OrientedPCurveEdge>],
    frame: Option<&super::super::plane_frame::PlaneFrame>,
) -> Point2 {
    // Hole polygons for SEED REJECTION must not UNDER-cover the true hole —
    // the opposite bias from `is_inside_any_hole`'s chord approximation. A
    // seed accepted in the sagitta gap between a corner arc's chord and the
    // arc itself lies inside the real hole and misclassifies the whole region
    // (the halfSockets clip cut: a 1.2 mm base ring whose inset corner arcs
    // have a ~0.75 mm sagitta — the ring floor classified Inside the tool and
    // the whole cut fell back to the mesh boolean). With the plane frame
    // available, densify curved edges by sampling their 3D curve (exact for
    // arcs; the stored pcurve can be a garbage-domain Nurbs2D, so it is never
    // sampled — see `hole_chord_polygon`) and projecting into UV.
    let hole_polys: Vec<Vec<Point2>> = inner_wires
        .iter()
        .map(|hole| {
            let mut pts: Vec<Point2> = Vec::with_capacity(hole.len() * 4);
            for e in hole {
                // With a frame available, derive EVERY vertex from 3D: the
                // stored start_uv can have been fitted in a DIFFERENT plane
                // frame (the same trap the curved-edge sampling below avoids
                // for pcurves), and one foreign-frame vertex corrupts the
                // whole rejection polygon — the even-odd test then accepts
                // seeds inside the hole (the seam-edge flush pocket dropped
                // the entire slab top this way).
                pts.push(match frame {
                    Some(f) => f.project(e.start_3d),
                    None => e.start_uv,
                });
                if matches!(e.curve_3d, brepkit_topology::edge::EdgeCurve::Line) {
                    continue;
                }
                if let Some(f) = frame {
                    // Sample in the edge's NATIVE orientation: a wire that
                    // traverses the edge reversed carries swapped
                    // start_3d/end_3d, and `domain_with_endpoints` always
                    // takes the positive parametric span — swapped endpoints
                    // would select the COMPLEMENTARY arc. Restore wire order
                    // afterwards so the polygon winding stays consistent.
                    let (s3, e3) = if e.forward {
                        (e.start_3d, e.end_3d)
                    } else {
                        (e.end_3d, e.start_3d)
                    };
                    let (t0, t1) = e.curve_3d.domain_with_endpoints(s3, e3);
                    // Dense enough for a SINGLE-edge closed hole (a full bore
                    // circle): 3 interior samples inscribe a square whose
                    // 0.29r sagitta gap accepts seeds well inside the hole.
                    let mut samples: Vec<Point2> = (1..16)
                        .map(|k| {
                            let t = (t1 - t0).mul_add(f64::from(k) / 16.0, t0);
                            f.project(e.curve_3d.evaluate_with_endpoints(t, s3, e3))
                        })
                        .collect();
                    if !e.forward {
                        samples.reverse();
                    }
                    pts.extend(samples);
                } else {
                    // No frame (non-planar surface): keep the chord-midpoint
                    // densification (the historical behavior).
                    pts.push(Point2::new(
                        0.5 * (e.start_uv.x() + e.end_uv.x()),
                        0.5 * (e.start_uv.y() + e.end_uv.y()),
                    ));
                }
            }
            pts
        })
        .collect();
    // With the frame, a hole of lines and conic arcs is read exactly: its
    // sampled polygon still sags inside each arc, and a ring thinner than
    // that sag (a ball touching only a frustum's cap) would take a seed in
    // the hole for material.
    let exact_holes: Vec<Option<Vec<Boundary2>>> = inner_wires
        .iter()
        .map(|hole| frame.and_then(|f| exact_hole(hole, f)))
        .collect();
    let in_any_hole = |pt: Point2| -> bool {
        hole_polys
            .iter()
            .zip(&exact_holes)
            .any(|(poly, exact)| match exact {
                Some(pieces) => point_in_region(pieces, pt, SEED_CLEARANCE) != Some(false),
                None => poly.len() >= 3 && super::super::classify_2d::point_in_polygon_2d(pt, poly),
            })
    };

    // Strategy: take midpoints between outer wire edge midpoints and the outer
    // boundary -- these are likely in the ring region between outer and inner.
    let centroid_x = outer_pts.iter().map(|p| p.x()).sum::<f64>() / outer_pts.len() as f64;
    let centroid_y = outer_pts.iter().map(|p| p.y()).sum::<f64>() / outer_pts.len() as f64;
    // Step inward from a boundary point toward the centroid in small
    // increments; the first point inside the outer wire and outside every
    // hole wins. Small steps handle THIN rings (e.g. the ~1.2mm gridfinity
    // lip annulus on an 83mm cap), where a single large nudge overshoots
    // straight into the hole and no ring point is ever found.
    let step = |from: Point2, t: f64| {
        Point2::new(
            from.x() * (1.0 - t) + centroid_x * t,
            from.y() * (1.0 - t) + centroid_y * t,
        )
    };
    let usable = |candidate: &Point2| {
        super::super::classify_2d::point_in_polygon_2d(*candidate, outer_pts)
            && !in_any_hole(*candidate)
    };
    let step_in = |from: Point2| {
        (1..=99)
            .map(|k| step(from, f64::from(k) * 0.005))
            .find(usable)
    };
    // A ring thinner than the first step (a ball touching a frustum's cap
    // leaves one a thousandth of its rim's radius wide) is stepped over by
    // every one of them; only then are finer steps tried.
    let step_in_finely = |from: Point2| {
        [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
            .into_iter()
            .map(|t| step(from, t))
            .find(usable)
    };
    let edge_mids = (0..outer_pts.len()).map(|i| {
        let j = (i + 1) % outer_pts.len();
        Point2::new(
            (outer_pts[i].x() + outer_pts[j].x()) * 0.5,
            (outer_pts[i].y() + outer_pts[j].y()) * 0.5,
        )
    });
    // The edge midpoints first. A curved rim reaches the outer polygon as
    // chords whose midpoints stand a sagitta inside it, so a ring thinner
    // than that (a hole near a circular cap's rim) has every midpoint in its
    // hole; the vertices, tried next, lie on the rim.
    if let Some(candidate) = edge_mids
        .clone()
        .chain(outer_pts.iter().copied())
        .find_map(step_in)
        .or_else(|| {
            edge_mids
                .chain(outer_pts.iter().copied())
                .find_map(step_in_finely)
        })
    {
        return candidate;
    }

    // Fallback: try vertex midpoints between consecutive outer wire vertices.
    if outer_pts.len() >= 2 {
        let mid = Point2::new(
            (outer_pts[0].x() + outer_pts[1].x()) * 0.5,
            (outer_pts[0].y() + outer_pts[1].y()) * 0.5,
        );
        return mid;
    }

    // Ultimate fallback: centroid (even though it may be in a hole).
    let n = outer_pts.len() as f64;
    Point2::new(
        outer_pts.iter().map(|p| p.x()).sum::<f64>() / n,
        outer_pts.iter().map(|p| p.y()).sum::<f64>() / n,
    )
}

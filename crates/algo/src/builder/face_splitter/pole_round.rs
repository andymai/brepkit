//! A ball face that holds one pole of its chart, its other pole off the
//! face, and the loops that split it.
//!
//! In the chart's `(u, v)`, a loop that winds the axis bounds no area, so
//! its pieces cannot be read there. Seen from the pole the face holds, with
//! the other pole off the face, the face is a region of a plane: the
//! azimuthal equal-area map about the held pole takes the whole ball, less
//! the other pole, onto a disc. There each loop is a closed curve that
//! winds as an outline or as a hole, and holes nest in their pieces.

use brepkit_math::madd::Madd;
use brepkit_math::surfaces::SphericalSurface;
use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::edge::EdgeCurve;

use super::super::classify_2d::point_in_polygon_2d;
use super::super::split_types::OrientedPCurveEdge;

/// Samples per curved edge.
const SAMPLES: u32 = 64;

/// The plane about a ball's held pole: the azimuthal equal-area map, which
/// keeps areas (the whole ball maps onto a disc of radius 2, area `4 pi`).
pub(super) struct PoleView {
    center: Point3,
    radius: f64,
    north: bool,
    pole: Vec3,
    x: Vec3,
    y: Vec3,
}

impl PoleView {
    /// The view of `sphere` about its north pole, or its south pole when
    /// `north` is false.
    pub(super) fn new(sphere: &SphericalSurface, north: bool) -> Self {
        let z = sphere.z_axis();
        Self {
            center: sphere.center(),
            radius: sphere.radius(),
            north,
            pole: if north { z } else { -z },
            x: sphere.x_axis(),
            y: sphere.y_axis(),
        }
    }

    /// Where `p` maps. `None` within about 1.4e-6 rad of the far pole,
    /// which has no image.
    fn map(&self, p: Point3) -> Option<Point2> {
        let d = (p - self.center) * (1.0 / self.radius);
        let rise = 1.0 + d.dot(self.pole);
        if rise <= 1e-12 {
            return None;
        }
        let k = (2.0 / rise).sqrt();
        Some(Point2::new(k * d.dot(self.x), k * d.dot(self.y)))
    }

    /// Points on the ball, mapped.
    fn polygon(&self, pts: &[Point3]) -> Option<Vec<Point2>> {
        pts.iter().map(|&p| self.map(p)).collect()
    }

    /// Twice the signed area a run of edges adds to a loop it lies in: the
    /// shoelace terms of each edge's samples, its end included. The terms of
    /// an edge and of its twin, walked the other way on the same samples,
    /// cancel.
    fn twice_area(&self, edges: &[OrientedPCurveEdge]) -> Option<f64> {
        let mut sum = 0.0;
        for e in edges {
            let mut pts = Vec::new();
            walk_edge(e, &mut pts);
            pts.push(e.end_3d);
            let mapped = self.polygon(&pts)?;
            sum += mapped
                .windows(2)
                .map(|w| w[0].x() * w[1].y() - w[1].x() * w[0].y())
                .sum::<f64>();
        }
        Some(sum)
    }

    /// The signed area of a loop; 0 where it does not map.
    pub(super) fn loop_area(&self, lp: &[OrientedPCurveEdge]) -> f64 {
        self.twice_area(lp).map_or(0.0, |a| 0.5 * a)
    }

    /// Whether the face whose outline is `boundary` (its edges as they are
    /// walked) lies on the outline's left in the chart's `(u, v)`. The face
    /// lies inside its outline here, on the outline's left where the outline
    /// runs counter-clockwise. The map keeps the way round of `(u, v)` about
    /// the north pole, whose distance from the centre falls as `v` rises,
    /// and turns it about the south pole.
    pub(super) fn face_left_in_uv(&self, boundary: &[OrientedPCurveEdge]) -> Option<bool> {
        let area = self.twice_area(boundary)?;
        (area.abs() > 1e-9).then_some((area > 0.0) == self.north)
    }

    /// A loop of edges, each sampled along its own curve in the way the
    /// loop walks it, mapped.
    fn loop_polygon(&self, lp: &[OrientedPCurveEdge]) -> Option<Vec<Point2>> {
        let mut pts = Vec::new();
        for e in lp {
            walk_edge(e, &mut pts);
        }
        self.polygon(&pts)
    }
}

/// `e`'s start, then samples along its curve toward its end, in the way the
/// loop walks it.
fn walk_edge(e: &OrientedPCurveEdge, out: &mut Vec<Point3>) {
    use std::f64::consts::TAU;
    out.push(e.start_3d);
    if matches!(e.curve_3d, EdgeCurve::Line) {
        return;
    }
    let fraction = |k: u32| f64::from(k) / f64::from(SAMPLES);
    let closed = (e.start_3d - e.end_3d).length() < 1e-10;
    match &e.curve_3d {
        EdgeCurve::Circle(c) if closed => {
            let (a0, turn) = (c.project(e.start_3d), if e.forward { TAU } else { -TAU });
            out.extend((1..SAMPLES).map(|k| c.evaluate(turn.madd(fraction(k), a0))));
        }
        EdgeCurve::Ellipse(el) if closed => {
            let (a0, turn) = (el.project(e.start_3d), if e.forward { TAU } else { -TAU });
            out.extend((1..SAMPLES).map(|k| el.evaluate(turn.madd(fraction(k), a0))));
        }
        curve => {
            let (s3, e3) = if e.forward {
                (e.start_3d, e.end_3d)
            } else {
                (e.end_3d, e.start_3d)
            };
            let (t0, t1) = curve.domain_with_endpoints(s3, e3);
            let mut samples: Vec<Point3> = (1..SAMPLES)
                .map(|k| curve.evaluate_with_endpoints((t1 - t0).madd(fraction(k), t0), s3, e3))
                .collect();
            // A marched curve's domain can run from either end.
            let at_t0 = curve.evaluate_with_endpoints(t0, s3, e3);
            if (at_t0 - e.start_3d).length() > (at_t0 - e.end_3d).length() {
                samples.reverse();
            }
            out.extend(samples);
        }
    }
}

/// The 3D point half way along `e`'s curve.
fn edge_middle(e: &OrientedPCurveEdge) -> Point3 {
    let (s3, e3) = if e.forward {
        (e.start_3d, e.end_3d)
    } else {
        (e.end_3d, e.start_3d)
    };
    let (t0, t1) = e.curve_3d.domain_with_endpoints(s3, e3);
    e.curve_3d
        .evaluate_with_endpoints(f64::midpoint(t0, t1), s3, e3)
}

/// Whether two loop edges are one edge of the face, in either sense.
fn same_edge(a: &OrientedPCurveEdge, b: &OrientedPCurveEdge, near: f64) -> bool {
    let ends = ((a.start_3d - b.start_3d).length() <= near
        && (a.end_3d - b.end_3d).length() <= near)
        || ((a.start_3d - b.end_3d).length() <= near && (a.end_3d - b.start_3d).length() <= near);
    ends && (edge_middle(a) - edge_middle(b)).length() <= near
}

/// A piece: its outline and its holes.
pub(super) type Piece = (Vec<OrientedPCurveEdge>, Vec<Vec<OrientedPCurveEdge>>);

/// The pieces that `loops` bound on a ball face whose outline is the run of
/// edges `boundary`, cut by the section halves `sections` (each section run
/// both ways), seen in `view`.
///
/// The loops must run every piece of the outline and every section half
/// once: a section the walk left out would leave the face standing whole
/// across it.
///
/// A loop that winds as the outline does is a piece's outline, one that
/// winds against it a hole. Each hole goes to the smallest outline that
/// holds the middle of its first edge, of those that do not run along that
/// edge (a hole's twin, the outline of the piece inside it). The areas
/// must balance: the pieces' outlines less their holes cover the face's
/// own area. Read on the same samples, each section the pieces run both
/// ways cancels and each piece of the outline counts once, so a piece left
/// out or taken twice shows.
///
/// `None` when a loop does not map, bounds no area, runs an edge both ways
/// or passes a vertex twice (two regions toured as one, touching there), a
/// hole finds no outline, or the areas do not balance.
pub(super) fn pieces(
    view: &PoleView,
    boundary: &[OrientedPCurveEdge],
    sections: &[OrientedPCurveEdge],
    loops: Vec<Vec<OrientedPCurveEdge>>,
    tol: f64,
) -> Option<Vec<Piece>> {
    let face_area = 0.5 * view.twice_area(boundary)?;
    if face_area.abs() <= 1e-9 {
        return None;
    }
    let runs = |on_section: bool| {
        loops
            .iter()
            .flatten()
            .filter(|e| e.source_edge_idx.is_some() == on_section)
            .count()
    };
    if sections.is_empty() || runs(true) != sections.len() || runs(false) != boundary.len() {
        return None;
    }
    let near = 100.0 * tol;
    let mut outers: Vec<(Vec<OrientedPCurveEdge>, Vec<Point2>, f64)> = Vec::new();
    let mut holes: Vec<(Vec<OrientedPCurveEdge>, f64)> = Vec::new();
    for lp in loops {
        let both_ways = lp
            .iter()
            .enumerate()
            .any(|(i, a)| lp[i + 1..].iter().any(|b| same_edge(a, b, near)));
        let pinched = lp.iter().enumerate().any(|(i, a)| {
            lp[i + 1..]
                .iter()
                .any(|b| (a.start_3d - b.start_3d).length() <= near)
        });
        if both_ways || pinched {
            return None;
        }
        let poly = view.loop_polygon(&lp)?;
        let area = 0.5 * view.twice_area(&lp)? * face_area.signum();
        let perimeter: f64 = poly
            .iter()
            .zip(poly.iter().cycle().skip(1))
            .map(|(a, b)| (*b - *a).length())
            .sum();
        if area.abs() <= perimeter * tol {
            return None;
        }
        if area > 0.0 {
            outers.push((lp, poly, area));
        } else {
            holes.push((lp, -area));
        }
    }
    let covered = outers.iter().map(|o| o.2).sum::<f64>() - holes.iter().map(|h| h.1).sum::<f64>();
    if (covered - face_area.abs()).abs() > 1e-6 * face_area.abs() {
        return None;
    }
    let mut pieces: Vec<Piece> = outers
        .iter()
        .map(|(lp, _, _)| (lp.clone(), Vec::new()))
        .collect();
    for (hole, _) in holes {
        let first = hole.first()?;
        let probe = view.map(edge_middle(first))?;
        let owner = outers
            .iter()
            .enumerate()
            .filter(|(_, (lp, poly, _))| {
                !lp.iter().any(|e| same_edge(e, first, near)) && point_in_polygon_2d(probe, poly)
            })
            .min_by(|a, b| a.1.2.total_cmp(&b.1.2))
            .map(|(i, _)| i)?;
        pieces[owner].1.push(hole);
    }
    Some(pieces)
}

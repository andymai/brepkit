//! Face filling: create a smooth NURBS surface from boundary curves.
//!
//! Fills an N-sided boundary with a surface patch. For 4-sided boundaries,
//! uses Coons patch interpolation.

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::OperationsError;

/// Fill a 4-sided boundary with a Coons patch.
///
/// Given 4 boundary curves as polylines, each ending where the next begins,
/// creates a smooth bilinear NURBS surface that interpolates the boundaries.
///
/// The curves run round the boundary: bottom, right, top, left. Each may be
/// given either way along, so a wire's edges in its own order (top running
/// back to the left, left running down) and the bottom-to-top,
/// left-to-right reading give the same patch.
///
/// # Errors
///
/// Returns an error if fewer than 4 boundary curves are provided, if the
/// curves do not close into a loop, or if curves have mismatched lengths.
pub fn fill_coons_patch(
    topo: &mut Topology,
    curves: &[Vec<Point3>],
) -> Result<brepkit_topology::face::FaceId, OperationsError> {
    if curves.len() < 4 {
        return Err(OperationsError::InvalidInput {
            reason: format!(
                "Coons patch requires 4 boundary curves, got {}",
                curves.len()
            ),
        });
    }
    if curves[..4].iter().any(|c| c.len() < 2) {
        return Err(OperationsError::InvalidInput {
            reason: "boundary curves must have at least 2 points each".into(),
        });
    }

    let around = chain_loop(&curves[..4])?;
    let bottom = &around[0];
    let right = &around[1];
    let top: Vec<Point3> = around[2].iter().rev().copied().collect();
    let left: Vec<Point3> = around[3].iter().rev().copied().collect();

    let n_u = bottom.len();
    let n_v = right.len();

    if top.len() != n_u || left.len() != n_v {
        return Err(OperationsError::InvalidInput {
            reason: "Coons patch boundary curves must have consistent point counts".into(),
        });
    }

    // Build the Coons patch: P(u,v) = Lc(u,v) + Ld(u,v) - B(u,v)
    // where:
    //   Lc = (1-v)*bottom(u) + v*top(u)      (linear blend of u-curves)
    //   Ld = (1-u)*left(v) + u*right(v)       (linear blend of v-curves)
    //   B  = bilinear interpolation of corners
    let p00 = bottom[0];
    let p10 = bottom[n_u - 1];
    let p01 = top[0];
    let p11 = top[n_u - 1];

    // The control net's rows run along u, one per bottom sample.
    let mut control_points = Vec::with_capacity(n_u);
    let mut weights = Vec::with_capacity(n_u);
    for i in 0..n_u {
        #[allow(clippy::cast_precision_loss)]
        let u = i as f64 / (n_u - 1) as f64;

        let mut row = Vec::with_capacity(n_v);
        for j in 0..n_v {
            #[allow(clippy::cast_precision_loss)]
            let v = j as f64 / (n_v - 1) as f64;

            // Lc: linear blend along v
            let lc = blend(bottom[i], top[i], v);

            // Ld: linear blend along u
            let ld = blend(left[j], right[j], u);

            // B: bilinear of corners
            let b = bilinear(p00, p10, p01, p11, u, v);

            // Coons: Lc + Ld - B
            row.push(Point3::new(
                lc.x() + ld.x() - b.x(),
                lc.y() + ld.y() - b.y(),
                lc.z() + ld.z() - b.z(),
            ));
        }

        control_points.push(row);
        weights.push(vec![1.0; n_v]);
    }

    // Build NURBS surface (degree 1 for bilinear Coons)
    let degree_u = 1.min(n_u - 1);
    let degree_v = 1.min(n_v - 1);

    let knots_u = build_clamped_knots(n_u, degree_u);
    let knots_v = build_clamped_knots(n_v, degree_v);

    let surface = NurbsSurface::new(
        degree_u,
        degree_v,
        knots_u,
        knots_v,
        control_points,
        weights,
    )?;

    let corners = [p00, p10, p11, p01];
    let verts: Vec<_> = corners
        .iter()
        .map(|&p| topo.add_vertex(Vertex::new(p, 1e-7)))
        .collect();

    // Each edge runs along its boundary curve, which the patch's boundary
    // follows sample for sample.
    let mut edges = Vec::with_capacity(4);
    for (i, curve) in around.iter().enumerate() {
        let next = (i + 1) % 4;
        let geometry = if curve.len() == 2 {
            EdgeCurve::Line
        } else {
            let mut points = curve.clone();
            let last = points.len() - 1;
            (points[0], points[last]) = (corners[i], corners[next]);
            EdgeCurve::NurbsCurve(
                NurbsCurve::new(
                    1,
                    build_clamped_knots(points.len(), 1),
                    points,
                    vec![1.0; curve.len()],
                )
                .map_err(OperationsError::Math)?,
            )
        };
        edges.push(topo.add_edge(Edge::new(verts[i], verts[next], geometry)));
    }

    let oriented: Vec<_> = edges
        .iter()
        .map(|&eid| OrientedEdge::new(eid, true))
        .collect();
    let wire = Wire::new(oriented, true).map_err(OperationsError::Topology)?;
    let wire_id = topo.add_wire(wire);

    let face = Face::new(wire_id, vec![], FaceSurface::Nurbs(surface));
    Ok(topo.add_face(face))
}

/// The four curves turned to run end to end round the loop the first two
/// start: each after the first begins where the one before ends.
fn chain_loop(curves: &[Vec<Point3>]) -> Result<Vec<Vec<Point3>>, OperationsError> {
    let ends = |c: &[Point3]| (c[0], c[c.len() - 1]);
    let gap = |c: &[Point3], at: Point3| {
        let (first, last) = ends(c);
        (first - at).length().min((last - at).length())
    };
    let size = curves.iter().flatten().fold(0.0_f64, |m, p| {
        m.max(p.x().abs()).max(p.y().abs()).max(p.z().abs())
    });
    let reach = 1e-6 * (1.0 + size);
    let mut first = curves[0].clone();
    if gap(&curves[1], first[0]) < gap(&curves[1], ends(&first).1) {
        first.reverse();
    }
    let mut around = vec![first];
    for curve in &curves[1..] {
        let at = ends(&around[around.len() - 1]).1;
        let (start, end) = ends(curve);
        let mut next = curve.clone();
        if (end - at).length() < (start - at).length() {
            next.reverse();
        }
        if (next[0] - at).length() > reach {
            return Err(OperationsError::InvalidInput {
                reason: "Coons patch boundary curves do not meet end to end".into(),
            });
        }
        around.push(next);
    }
    if (ends(&around[3]).1 - around[0][0]).length() > reach {
        return Err(OperationsError::InvalidInput {
            reason: "Coons patch boundary curves do not close into a loop".into(),
        });
    }
    Ok(around)
}

/// Linear blend: (1-t)*a + t*b
fn blend(a: Point3, b: Point3, t: f64) -> Point3 {
    Point3::new(
        a.x().mul_add(1.0 - t, b.x() * t),
        a.y().mul_add(1.0 - t, b.y() * t),
        a.z().mul_add(1.0 - t, b.z() * t),
    )
}

/// Bilinear interpolation of 4 corner points.
fn bilinear(p00: Point3, p10: Point3, p01: Point3, p11: Point3, u: f64, v: f64) -> Point3 {
    let bottom = blend(p00, p10, u);
    let top = blend(p01, p11, u);
    blend(bottom, top, v)
}

/// Build a clamped knot vector for n control points and given degree.
fn build_clamped_knots(n: usize, degree: usize) -> Vec<f64> {
    let mut knots = Vec::with_capacity(n + degree + 1);
    knots.extend(std::iter::repeat_n(0.0, degree + 1));
    let internal = n.saturating_sub(degree + 1);
    for i in 1..=internal {
        #[allow(clippy::cast_precision_loss)]
        knots.push(i as f64 / (internal + 1) as f64);
    }
    knots.extend(std::iter::repeat_n(1.0, degree + 1));
    knots
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn coons_patch_flat_square() {
        let mut topo = Topology::new();

        // 4 boundary curves forming a unit square in z=0
        let bottom = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)];
        let right = vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 0.0)];
        let top = vec![Point3::new(0.0, 1.0, 0.0), Point3::new(1.0, 1.0, 0.0)];
        let left = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)];

        let face_id = fill_coons_patch(&mut topo, &[bottom, right, top, left]).unwrap();

        let face = topo.face(face_id).unwrap();
        assert!(matches!(face.surface(), FaceSurface::Nurbs(_)));
    }

    #[test]
    fn coons_patch_saddle() {
        let mut topo = Topology::new();

        // Saddle-shaped boundary (corners at different heights)
        let bottom = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.5, 0.0, 0.1),
            Point3::new(1.0, 0.0, 0.0),
        ];
        let right = vec![
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 0.5, 0.1),
            Point3::new(1.0, 1.0, 0.0),
        ];
        let top = vec![
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.5, 1.0, -0.1),
            Point3::new(1.0, 1.0, 0.0),
        ];
        let left = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 0.5, -0.1),
            Point3::new(0.0, 1.0, 0.0),
        ];

        let face_id = fill_coons_patch(&mut topo, &[bottom, right, top, left]).unwrap();

        let face = topo.face(face_id).unwrap();
        if let FaceSurface::Nurbs(surf) = face.surface() {
            // Center of saddle should be near (0.5, 0.5, ~0)
            let center = surf.evaluate(0.5, 0.5);
            assert!((center.x() - 0.5).abs() < 0.2);
            assert!((center.y() - 0.5).abs() < 0.2);
        }
    }

    /// The issue's patch over four lines, (0, 0, 0), (10, 0, 0), (10, 10, 5)
    /// and (0, 10, 0): its curves taken in the wire's order or read bottom
    /// to top and left to right give the surface z = x y / 20, of area
    /// 107.904.
    #[test]
    fn coons_patch_takes_its_curves_either_way_along() {
        let (a, b, c, d) = (
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
            Point3::new(10.0, 10.0, 5.0),
            Point3::new(0.0, 10.0, 0.0),
        );
        for curves in [
            vec![vec![a, b], vec![b, c], vec![c, d], vec![d, a]],
            vec![vec![a, b], vec![b, c], vec![d, c], vec![a, d]],
            vec![vec![b, a], vec![c, b], vec![c, d], vec![a, d]],
        ] {
            let mut topo = Topology::new();
            let face = fill_coons_patch(&mut topo, &curves).unwrap();
            let surface = topo.face(face).unwrap().surface();
            assert!(matches!(surface, FaceSurface::Nurbs(_)));
            if let FaceSurface::Nurbs(surf) = surface {
                let along = surf.evaluate(0.25, 0.0);
                assert!(
                    (along - Point3::new(2.5, 0.0, 0.0)).length() < 1e-12,
                    "{along:?}"
                );
                for (u, v) in [(0.25, 0.5), (0.5, 0.75), (0.9, 0.1)] {
                    let p = surf.evaluate(u, v);
                    assert!(
                        (p.z() - p.x() * p.y() / 20.0).abs() < 1e-12,
                        "{curves:?}: {p:?}"
                    );
                }
            }
            let area = crate::measure::face_area(&topo, face, 0.01).unwrap();
            assert!(
                (area - 107.904).abs() < 1e-3 * 107.904,
                "{curves:?}: area {area}"
            );
        }
    }

    /// A boundary curve given as samples of an arc bounds the patch along
    /// those samples, and its edge runs along them too, not along the chord
    /// between its ends.
    #[test]
    fn coons_patch_edges_follow_their_curves() {
        let arc: Vec<Point3> = (0..=10)
            .map(|i| {
                let t = f64::from(i) / 10.0;
                Point3::new(10.0 * t, 0.0, 3.0 * (std::f64::consts::PI * t).sin())
            })
            .collect();
        let right = vec![Point3::new(10.0, 0.0, 0.0), Point3::new(10.0, 10.0, 0.0)];
        let top: Vec<Point3> = (0..=10)
            .map(|i| Point3::new(10.0 - f64::from(i), 10.0, 0.0))
            .collect();
        let left = vec![Point3::new(0.0, 10.0, 0.0), Point3::new(0.0, 0.0, 0.0)];
        let mut topo = Topology::new();
        let face = fill_coons_patch(&mut topo, &[arc, right, top, left]).unwrap();
        let wire = topo.wire(topo.face(face).unwrap().outer_wire()).unwrap();
        let bottom = topo.edge(wire.edges()[0].edge()).unwrap();
        let (p, q) = (
            topo.vertex(bottom.start()).unwrap().point(),
            topo.vertex(bottom.end()).unwrap().point(),
        );
        let (t0, t1) = bottom.curve().domain_with_endpoints(p, q);
        let mid = bottom
            .curve()
            .evaluate_with_endpoints(0.5 * (t0 + t1), p, q);
        assert!(
            (mid - Point3::new(5.0, 0.0, 3.0)).length() < 1e-12,
            "{mid:?}"
        );
    }

    #[test]
    fn coons_patch_refuses_curves_that_do_not_close() {
        let mut topo = Topology::new();
        let p = |x: f64, y: f64| Point3::new(x, y, 0.0);
        let curves = [
            vec![p(0.0, 0.0), p(1.0, 0.0)],
            vec![p(1.0, 0.0), p(1.0, 1.0)],
            vec![p(1.0, 1.0), p(0.0, 1.0)],
            vec![p(0.0, 1.0), p(0.0, 0.5)],
        ];
        assert!(fill_coons_patch(&mut topo, &curves).is_err());
    }

    #[test]
    fn coons_patch_too_few_curves_error() {
        let mut topo = Topology::new();
        let curve = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)];
        assert!(fill_coons_patch(&mut topo, &[curve]).is_err());
    }

    #[test]
    fn coons_patch_mismatched_lengths_error() {
        let mut topo = Topology::new();
        let bottom = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)];
        let right = vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 0.0)];
        let top = vec![
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
        ]; // 3 points vs bottom's 2
        let left = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)];

        assert!(fill_coons_patch(&mut topo, &[bottom, right, top, left]).is_err());
    }
}

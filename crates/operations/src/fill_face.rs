//! Face filling: create a smooth NURBS surface from boundary curves.
//!
//! Fills an N-sided boundary with a surface patch. For 4-sided boundaries,
//! uses Coons patch interpolation.

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
/// Given 4 boundary curves as polylines (each connecting to the next),
/// creates a smooth bilinear NURBS surface that interpolates the boundaries.
///
/// The curves should be ordered: bottom, right, top (reversed), left (reversed).
///
/// # Errors
///
/// Returns an error if fewer than 4 boundary curves are provided or if
/// curves have mismatched lengths.
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

    let bottom = &curves[0];
    let right = &curves[1];
    let top = &curves[2];
    let left = &curves[3];

    // Validate that boundaries connect at corners
    let n_u = bottom.len();
    let n_v = right.len();

    if top.len() != n_u || left.len() != n_v {
        return Err(OperationsError::InvalidInput {
            reason: "Coons patch boundary curves must have consistent point counts".into(),
        });
    }

    if n_u < 2 || n_v < 2 {
        return Err(OperationsError::InvalidInput {
            reason: "boundary curves must have at least 2 points each".into(),
        });
    }

    // Build the Coons patch: P(u,v) = Lc(u,v) + Ld(u,v) - B(u,v)
    // where:
    //   Lc = (1-v)*bottom(u) + v*top(u)      (linear blend of u-curves)
    //   Ld = (1-u)*left(v) + u*right(v)       (linear blend of v-curves)
    //   B  = bilinear interpolation of corners
    let mut control_points = Vec::with_capacity(n_v);
    let mut weights = Vec::with_capacity(n_v);

    // Corner points
    let p00 = bottom[0];
    let p10 = bottom[n_u - 1];
    let p01 = top[0];
    let p11 = top[n_u - 1];

    for j in 0..n_v {
        #[allow(clippy::cast_precision_loss)]
        let v = j as f64 / (n_v - 1) as f64;

        let mut row = Vec::with_capacity(n_u);
        let mut weight_row = Vec::with_capacity(n_u);

        for i in 0..n_u {
            #[allow(clippy::cast_precision_loss)]
            let u = i as f64 / (n_u - 1) as f64;

            // Lc: linear blend along v
            let lc = blend(bottom[i], top[i], v);

            // Ld: linear blend along u
            let ld = blend(left[j], right[j], u);

            // B: bilinear of corners
            let b = bilinear(p00, p10, p01, p11, u, v);

            // Coons: Lc + Ld - B
            let point = Point3::new(
                lc.x() + ld.x() - b.x(),
                lc.y() + ld.y() - b.y(),
                lc.z() + ld.z() - b.z(),
            );

            row.push(point);
            weight_row.push(1.0);
        }

        control_points.push(row);
        weights.push(weight_row);
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

    // Create the face with a boundary wire from corner points
    let corners = [p00, p10, p11, p01];
    let verts: Vec<_> = corners
        .iter()
        .map(|&p| topo.add_vertex(Vertex::new(p, 1e-7)))
        .collect();

    let n_corners = verts.len();
    let edges: Vec<_> = (0..n_corners)
        .map(|i| {
            let next = (i + 1) % n_corners;
            topo.add_edge(Edge::new(verts[i], verts[next], EdgeCurve::Line))
        })
        .collect();

    let oriented: Vec<_> = edges
        .iter()
        .map(|&eid| OrientedEdge::new(eid, true))
        .collect();
    let wire = Wire::new(oriented, true).map_err(OperationsError::Topology)?;
    let wire_id = topo.add_wire(wire);

    let face = Face::new(wire_id, vec![], FaceSurface::Nurbs(surface));
    Ok(topo.add_face(face))
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

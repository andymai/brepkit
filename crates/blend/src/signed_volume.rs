//! Signed shell volume — orientation sign detection for blended results.
//!
//! The blend crate cannot depend on `brepkit-operations` (dependency cycle),
//! so this module provides a small, sign-accurate approximation of the
//! oriented volume of a set of faces forming a closed shell. Only the SIGN
//! is load-bearing: the builder uses it to detect a globally inverted shell
//! and flip every face's `reversed` flag so the published solid is outward.
//!
//! Planar faces use the exact divergence-theorem contribution
//! `(1/3)·d_out·A` with `d_out = ±d` from the face's `reversed` flag and
//! `A = |outer loop| − Σ|hole loops|` (holes subtract by magnitude, matching
//! the measurement crate's convention — boolean results can emit hole rims
//! wound either way). Analytic and NURBS faces use a coarse UV grid over
//! their full parameter domain; the cell triangles follow the
//! parameterization's natural orientation, which coincides with the surface
//! normal for all built-in surfaces, and the sum is multiplied by the
//! `reversed` sign. Because the integral is homogeneous in the domain, a
//! partial patch or a hole changes only the magnitude, never the sign.

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};

/// Estimate the signed volume of the shell formed by `faces`.
///
/// Negative when the shell is inverted (inward-pointing normals), positive
/// when outward. The magnitude is only approximate and must not be used for
/// measurements.
///
/// # Errors
///
/// Returns a [`brepkit_topology::TopologyError`] on topology lookup failure.
pub fn signed_shell_volume(
    topo: &Topology,
    faces: &[FaceId],
) -> Result<f64, brepkit_topology::TopologyError> {
    let mut total = 0.0;
    for &face_id in faces {
        let face = topo.face(face_id)?;
        let sign = if face.is_reversed() { -1.0 } else { 1.0 };
        total += match face.surface() {
            FaceSurface::Plane { normal, d } => {
                planar_face_contribution(topo, face_id, *normal, *d, sign)?
            }
            other => surface_grid_contribution(other, sign),
        };
    }
    Ok(total)
}

fn planar_face_contribution(
    topo: &Topology,
    face_id: FaceId,
    normal: Vec3,
    d: f64,
    sign: f64,
) -> Result<f64, brepkit_topology::TopologyError> {
    let face = topo.face(face_id)?;
    // Right-handed in-plane frame: ex × ey = normal, so a loop wound CCW as
    // seen from +normal yields a positive signed area.
    let frame = match brepkit_math::frame::Frame3::from_normal(Point3::new(0.0, 0.0, 0.0), normal) {
        Ok(frame) => frame,
        Err(_) => return Ok(0.0),
    };
    let (ex, ey) = (frame.x, frame.y);

    let wire_area =
        |wire_id: brepkit_topology::wire::WireId| -> Result<f64, brepkit_topology::TopologyError> {
            let wire = topo.wire(wire_id)?;
            let mut area2 = 0.0;
            for oriented in wire.edges() {
                let edge = topo.edge(oriented.edge())?;
                let start = topo.vertex(edge.start())?.point();
                let end = topo.vertex(edge.end())?.point();
                // Green's theorem doubled signed area: ∮ (x dy − y dx) = Σ (Sx·Ey − Sy·Ex).
                // A reversed oriented edge is traversed from end to start.
                let (ax, ay) = (
                    start.x() * ex.x() + start.y() * ex.y() + start.z() * ex.z(),
                    start.x() * ey.x() + start.y() * ey.y() + start.z() * ey.z(),
                );
                let (bx, by) = (
                    end.x() * ex.x() + end.y() * ex.y() + end.z() * ex.z(),
                    end.x() * ey.x() + end.y() * ey.y() + end.z() * ey.z(),
                );
                let (ax, ay, bx, by) = if oriented.is_forward() {
                    (ax, ay, bx, by)
                } else {
                    (bx, by, ax, ay)
                };
                area2 += ax * by - ay * bx;
            }
            Ok(area2)
        };

    let outer_area2 = wire_area(face.outer_wire())?.abs();
    let mut hole_area2 = 0.0;
    for &inner in face.inner_wires() {
        hole_area2 += wire_area(inner)?.abs();
    }
    let area = (outer_area2 - hole_area2).max(0.0) / 2.0;
    // Divergence-theorem plane contribution: (1/3)·d_out·A with
    // d_out = sign·d (sign already encodes `is_reversed`).
    Ok(sign * d * area / 3.0)
}

fn surface_grid_contribution(surface: &FaceSurface, sign: f64) -> f64 {
    let (umin, umax, vmin, vmax) = match surface {
        FaceSurface::Plane { .. } => return 0.0,
        FaceSurface::Cylinder(_) => (0.0, std::f64::consts::TAU, 0.0, 1.0),
        FaceSurface::Cone(_) => (0.0, std::f64::consts::TAU, 0.0, 1.0),
        FaceSurface::Sphere(_) => (
            0.0,
            std::f64::consts::TAU,
            -std::f64::consts::FRAC_PI_2,
            std::f64::consts::FRAC_PI_2,
        ),
        FaceSurface::Torus(_) => (0.0, std::f64::consts::TAU, 0.0, std::f64::consts::TAU),
        FaceSurface::Nurbs(n) => (
            n.domain_u().0,
            n.domain_u().1,
            n.domain_v().0,
            n.domain_v().1,
        ),
    };
    let mut volume = 0.0;
    let cell_count = 16;
    for ui in 0..cell_count {
        for vi in 0..cell_count {
            let u0 = umin + (umax - umin) * (ui as f64) / cell_count as f64;
            let u1 = umin + (umax - umin) * ((ui + 1) as f64) / cell_count as f64;
            let v0 = vmin + (vmax - vmin) * (vi as f64) / cell_count as f64;
            let v1 = vmin + (vmax - vmin) * ((vi + 1) as f64) / cell_count as f64;
            let a = surface
                .evaluate(u0, v0)
                .unwrap_or(Point3::new(0.0, 0.0, 0.0));
            let b = surface
                .evaluate(u1, v0)
                .unwrap_or(Point3::new(0.0, 0.0, 0.0));
            let c = surface
                .evaluate(u1, v1)
                .unwrap_or(Point3::new(0.0, 0.0, 0.0));
            let d = surface
                .evaluate(u0, v1)
                .unwrap_or(Point3::new(0.0, 0.0, 0.0));
            volume += tetrahedron(a, b, c) * sign;
            volume += tetrahedron(a, c, d) * sign;
        }
    }
    volume
}

fn tetrahedron(a: Point3, b: Point3, c: Point3) -> f64 {
    let a = Vec3::new(a.x(), a.y(), a.z());
    let b = Vec3::new(b.x(), b.y(), b.z());
    let c = Vec3::new(c.x(), c.y(), c.z());
    1.0 / 6.0 * a.dot(b.cross(c))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use brepkit_math::surfaces::SphericalSurface;

    #[test]
    fn sphere_grid_flux_is_analytic_volume_and_translation_invariant() {
        // A closed analytic sphere's divergence flux equals (4/3)·π·r³ for any
        // center. The u/v bounds must match SphericalSurface (u = azimuth over
        // 0..2π, v = latitude over -π/2..π/2); swapping them mis-tiles the
        // sphere into a wrong, center-dependent value.
        let radius = 2.0_f64;
        let expected = 4.0 / 3.0 * std::f64::consts::PI * radius.powi(3);
        let flux = |center: Point3| {
            surface_grid_contribution(
                &FaceSurface::Sphere(SphericalSurface::new(center, radius).unwrap()),
                1.0,
            )
        };
        let at_origin = flux(Point3::new(0.0, 0.0, 0.0));
        let off_origin = flux(Point3::new(5.0, -3.0, 4.0));
        assert!(
            (at_origin - expected).abs() < 0.05 * expected,
            "sphere flux should be ~{expected}, got {at_origin}"
        );
        assert!(
            (off_origin - at_origin).abs() < 1e-6,
            "sphere flux must be translation-invariant: origin={at_origin}, offset={off_origin}"
        );
    }
}

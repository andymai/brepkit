//! Booleans of two spheres whose centres are offset: the result keeps its
//! spherical faces and the volume the lens between the spheres gives.

#![allow(clippy::unwrap_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_sphere;
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;

/// The volume two spheres share, their centres `d` apart.
fn lens(r1: f64, r2: f64, d: f64) -> f64 {
    PI * (r1 + r2 - d).powi(2)
        * (d * d + 2.0 * d * r2 - 3.0 * r2 * r2 + 2.0 * d * r1 + 6.0 * r1 * r2 - 3.0 * r1 * r1)
        / (12.0 * d)
}

fn check(offset: [f64; 3], segments: usize) {
    check_radii(1.5, offset, segments);
}

/// Booleans of a ball of radius 2 at the origin and one of radius `r2` moved
/// by `offset`.
fn check_radii(r2: f64, offset: [f64; 3], segments: usize) {
    let r1 = 2.0;
    let d = offset.iter().map(|c| c * c).sum::<f64>().sqrt();
    let shared = lens(r1, r2, d);
    let ball = |r: f64| 4.0 * PI * r * r * r / 3.0;
    for (op, want) in [
        (BooleanOp::Cut, ball(r1) - shared),
        (BooleanOp::Fuse, ball(r1) + ball(r2) - shared),
        (BooleanOp::Intersect, shared),
    ] {
        let mut topo = Topology::default();
        let a = make_sphere(&mut topo, r1, segments).unwrap();
        let b = make_sphere(&mut topo, r2, segments).unwrap();
        transform_solid(
            &mut topo,
            b,
            &Mat4::translation(offset[0], offset[1], offset[2]),
        )
        .unwrap();
        let result = boolean(&mut topo, op, a, b).unwrap();
        let faces = solid_faces(&topo, result).unwrap();
        assert!(
            faces
                .iter()
                .all(|&f| topo.face(f).unwrap().surface().type_tag() == "sphere"),
            "{op:?} at {offset:?} ({segments} segments): {} faces, not all spherical",
            faces.len()
        );
        assert!(validate_solid(&topo, result).unwrap().is_valid());
        let got = solid_volume(&topo, result, 0.005).unwrap();
        assert!(
            (got - want).abs() < 1e-4 * want,
            "{op:?} at {offset:?} ({segments} segments): volume {got} against {want}"
        );
    }
}

#[test]
fn offset_sphere_booleans_stay_spherical() {
    for segments in [16, 32] {
        // Circles crossing both spheres' equators, and one crossing neither.
        check([1.6, 0.3, 0.2], segments);
        check([2.2, 0.1, 0.7], segments);
        check([0.5, 0.4, -0.9], segments);
    }
}

/// Offsets in the plane z = 0 put both equators in one plane, so the circle
/// the spheres share meets both seams at the same two points.
#[test]
fn offset_spheres_sharing_an_equator_plane() {
    for segments in [16, 32] {
        check([1.2, 0.0, 0.0], segments);
        check([0.0, 1.2, 0.0], segments);
        check([-1.5, 0.7, 0.0], segments);
        // The circle meets the larger sphere's seam at 45 degrees, on a
        // vertex of its equator at either segment count.
        check([2.0f64.sqrt() + 0.5, 0.0, 0.0], segments);
        // Equal radii: the two seams' arcs between the crossings are both
        // under half a turn.
        check_radii(2.0, [1.2, 0.0, 0.0], segments);
        // A circle through the first ball's poles, which splits its
        // hemispheres into lunes.
        check_radii(1.2f64.hypot(2.0), [1.2, 0.0, 0.0], segments);
    }
}

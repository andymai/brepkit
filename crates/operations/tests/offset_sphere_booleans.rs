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
    let (r1, r2) = (2.0, 1.5);
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

#[test]
#[ignore = "both spheres' faceted equators lie in one plane; the booleans fall back to a mesh"]
fn offset_spheres_sharing_an_equator_plane() {
    check([1.2, 0.0, 0.0], 32);
}

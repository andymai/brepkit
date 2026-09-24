//! A non-uniform scale turns spheres and tori into exact NURBS images. The
//! ellipsoid primitive's hemispheres are caps bounded by the equator alone, a
//! boundary that winds the periodic direction once and is closed by a pole
//! row; a torus's one face closes on itself in both directions.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::measure::oriented_solid_volume;
use brepkit_operations::primitives::{make_cone, make_cylinder, make_sphere, make_torus};
use brepkit_operations::tessellate::{
    boundary_edge_count, non_manifold_edge_count, tessellate, tessellate_solid,
    tessellate_solid_with_tolerance,
};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;

#[test]
fn ellipsoid_meshes_watertight_at_its_volume() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut topo = Topology::new();
    let unit = make_sphere(&mut topo, 1.0, 16).unwrap();
    let unit_volume = oriented_solid_volume(&topo, unit, 0.001).unwrap();

    let ellipsoid = make_sphere(&mut topo, 1.0, 16).unwrap();
    transform_solid(&mut topo, ellipsoid, &Mat4::scale(2.0, 3.0, 4.0)).unwrap();
    for deflection in [0.01, 0.001] {
        let mesh = tessellate_solid(&topo, ellipsoid, deflection).unwrap();
        assert!(!mesh.indices.is_empty(), "empty mesh at {deflection}");
        assert_eq!(boundary_edge_count(&mesh), 0, "open mesh at {deflection}");
        assert_eq!(
            non_manifold_edge_count(&mesh),
            0,
            "non-manifold mesh at {deflection}"
        );
    }
    // The equator is a polygon of chords, so compare against the unit
    // sphere built the same way, scaled by the map's determinant.
    let volume = oriented_solid_volume(&topo, ellipsoid, 0.001).unwrap();
    let expected = 24.0 * unit_volume;
    assert!(
        (volume - expected).abs() < 5e-3 * expected,
        "ellipsoid volume {volume}, expected {expected}"
    );

    for fid in solid_faces(&topo, ellipsoid).unwrap() {
        let mesh = tessellate(&topo, fid, 0.01).unwrap();
        assert!(!mesh.indices.is_empty(), "face {fid:?} meshes on its own");
    }
}

#[test]
fn squashed_torus_meshes_watertight_at_its_volume() {
    let mut topo = Topology::new();
    let torus = make_torus(&mut topo, 5.0, 1.5, 16).unwrap();
    transform_solid(
        &mut topo,
        torus,
        &(Mat4::rotation_z(0.3) * Mat4::scale(2.0, 1.0, 1.0)),
    )
    .unwrap();
    for deflection in [0.01, 0.001] {
        let mesh = tessellate_solid(&topo, torus, deflection).unwrap();
        assert!(!mesh.indices.is_empty(), "empty mesh at {deflection}");
        assert_eq!(boundary_edge_count(&mesh), 0, "open mesh at {deflection}");
        assert_eq!(
            non_manifold_edge_count(&mesh),
            0,
            "non-manifold mesh at {deflection}"
        );
    }
    let exact = 2.0 * 2.0 * std::f64::consts::PI.powi(2) * 5.0 * 1.5 * 1.5;
    let volume = oriented_solid_volume(&topo, torus, 0.001).unwrap();
    assert!(
        (volume - exact).abs() < 1e-3 * exact,
        "torus volume {volume}, expected {exact}"
    );
}

/// A mirror keeps each hemisphere's NURBS boundary counter-clockwise about
/// its own (now inward) Su × Sv and flips the face's flag instead, so the
/// pole closure still finds the cap on the loop's left.
#[test]
fn mirrored_ellipsoid_meshes_watertight_at_its_volume() {
    let mut topo = Topology::new();
    let unit = make_sphere(&mut topo, 1.0, 16).unwrap();
    let unit_volume = oriented_solid_volume(&topo, unit, 0.001).unwrap();

    let ellipsoid = make_sphere(&mut topo, 1.0, 16).unwrap();
    transform_solid(&mut topo, ellipsoid, &Mat4::scale(-2.0, 3.0, 4.0)).unwrap();
    for deflection in [0.01, 0.001] {
        let mesh = tessellate_solid(&topo, ellipsoid, deflection).unwrap();
        assert!(!mesh.indices.is_empty(), "empty mesh at {deflection}");
        assert_eq!(boundary_edge_count(&mesh), 0, "open mesh at {deflection}");
        assert_eq!(
            non_manifold_edge_count(&mesh),
            0,
            "non-manifold mesh at {deflection}"
        );
    }
    let volume = oriented_solid_volume(&topo, ellipsoid, 0.001).unwrap();
    let expected = 24.0 * unit_volume;
    assert!(
        (volume - expected).abs() < 5e-3 * expected,
        "mirrored ellipsoid volume {volume}, expected {expected}"
    );
}

/// How far a wall triangle's centroid and edge midpoints stray from the
/// squashed cone `sqrt((x/2)² + y²) = r0 + k·z` (a cylinder at `k = 0`), read
/// in the frame the solid was squashed in. Triangles lying flat at one
/// height belong to the caps; every other triangle is the wall's, vertices
/// included. Also returns how many wall triangles were read.
fn max_wall_deviation(
    mesh: &brepkit_operations::tessellate::TriangleMesh,
    unrotate: Mat4,
    r0: f64,
    k: f64,
) -> (f64, usize) {
    let distance = |p: Point3| {
        let p = unrotate.mul_point(p);
        let q = (p.x() * p.x() / 4.0 + p.y() * p.y()).sqrt();
        let gradient = ((p.x() / (4.0 * q)).powi(2) + (p.y() / q).powi(2) + k * k).sqrt();
        k.mul_add(-p.z(), q - r0).abs() / gradient
    };
    let mid = |a: Point3, b: Point3| a + (b - a) * 0.5;
    let mut worst = 0.0_f64;
    let mut read = 0;
    for tri in mesh.indices.chunks_exact(3) {
        let [a, b, c] = [tri[0], tri[1], tri[2]].map(|i| mesh.positions[i as usize]);
        let flat = (a.z() - b.z()).abs() < 1e-9 && (a.z() - c.z()).abs() < 1e-9;
        if flat {
            continue;
        }
        read += 1;
        let centroid = a + ((b - a) + (c - a)) * (1.0 / 3.0);
        for p in [a, b, c, centroid, mid(a, b), mid(b, c), mid(c, a)] {
            worst = worst.max(distance(p));
        }
    }
    (worst, read)
}

/// A squashed cylinder's or frustum's wall is an exact NURBS elliptic cone.
/// Its interior grid follows the wall's own iso-line chords, so the mesh
/// stays within the deflection of the wall and converges on the closed-form
/// volume.
#[test]
fn squashed_walls_mesh_within_their_deflection() {
    let (r0, r1, h): (f64, f64, f64) = (1.5, 0.5, 4.0);
    let frustum = std::f64::consts::PI * h / 3.0 * r1.mul_add(r1, r0.mul_add(r0, r0 * r1));
    let cases = [
        (r0, 2.0 * std::f64::consts::PI * r0 * r0 * h),
        (r1, 2.0 * frustum),
    ];
    for (top, exact) in cases {
        let mut topo = Topology::new();
        let solid = if (top - r0).abs() < f64::EPSILON {
            make_cylinder(&mut topo, r0, h).unwrap()
        } else {
            make_cone(&mut topo, r0, top, h).unwrap()
        };
        let rotation = 0.3;
        transform_solid(
            &mut topo,
            solid,
            &(Mat4::rotation_z(rotation) * Mat4::scale(2.0, 1.0, 1.0)),
        )
        .unwrap();
        let k = (top - r0) / h;
        for (deflection, bound) in [(0.01, 3e-3), (0.001, 2e-4)] {
            let mesh = tessellate_solid(&topo, solid, deflection).unwrap();
            assert_eq!(boundary_edge_count(&mesh), 0, "open mesh at {deflection}");
            assert_eq!(
                non_manifold_edge_count(&mesh),
                0,
                "non-manifold mesh at {deflection}"
            );
            let (deviation, read) = max_wall_deviation(&mesh, Mat4::rotation_z(-rotation), r0, k);
            assert!(read > 100, "top {top}: only {read} wall triangles");
            assert!(
                deviation <= deflection,
                "top {top}: wall strays {deviation} at {deflection}"
            );
            let volume = oriented_solid_volume(&topo, solid, deflection).unwrap();
            assert!(
                (volume - exact).abs() < bound * exact,
                "top {top}: volume {volume} at {deflection}, exact {exact}"
            );
        }
    }
}

/// A small squashed torus meshed with a deflection far coarser than the
/// torus and a strict angular tolerance: the grid follows the angle, so no
/// mesh edge joins normals turned by more than a cell's diagonal allows.
#[test]
fn small_squashed_torus_follows_its_angular_tolerance() {
    let mut topo = Topology::new();
    let torus = make_torus(&mut topo, 1.0, 0.3, 16).unwrap();
    transform_solid(&mut topo, torus, &Mat4::scale(2.0, 1.0, 1.0)).unwrap();
    let angular = 0.2;
    let mesh = tessellate_solid_with_tolerance(&topo, torus, 1.0, angular).unwrap();
    assert_eq!(boundary_edge_count(&mesh), 0, "open mesh");
    let mut worst = 0.0_f64;
    for tri in mesh.indices.chunks_exact(3) {
        for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let (na, nb) = (mesh.normals[a as usize], mesh.normals[b as usize]);
            worst = worst.max(na.dot(nb).clamp(-1.0, 1.0).acos());
        }
    }
    assert!(worst <= 2.0 * angular, "mesh edge turns {worst}");
}

//! A ball cut out of a box leaves its sphere faces reversed: the builder flips
//! the flag and keeps the wire, which runs about the sphere's outward normal.
//! A pocket bounded by one circle (a dimple) meshes over its own side of the
//! circle, a cavity's hemispheres each over their own half, and a dimpled
//! box scaled unevenly keeps the dimple it had.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::{face_area, oriented_solid_volume, solid_volume};
use brepkit_operations::primitives::{make_box, make_sphere};
use brepkit_operations::tessellate::{is_watertight, tessellate, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

/// The box `[-5, 5]³` less a ball of radius 2 centred at `(0, 0, z)`.
fn box_less_ball(z: f64) -> (Topology, SolidId) {
    let mut topo = Topology::new();
    let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    transform_solid(&mut topo, block, &Mat4::translation(-5.0, -5.0, -5.0)).unwrap();
    let ball = make_sphere(&mut topo, 2.0, 32).unwrap();
    transform_solid(&mut topo, ball, &Mat4::translation(0.0, 0.0, z)).unwrap();
    let piece = boolean(&mut topo, BooleanOp::Cut, block, ball).unwrap();
    (topo, piece)
}

fn mesh_area(topo: &Topology, face: brepkit_topology::face::FaceId) -> f64 {
    let mesh = tessellate(topo, face, 0.005).unwrap();
    mesh.indices
        .chunks_exact(3)
        .map(|t| {
            let [a, b, c] = [0, 1, 2].map(|k| mesh.positions[t[k] as usize]);
            (b - a).cross(c - a).length() / 2.0
        })
        .sum()
}

/// A ball of radius 2 at `(0, 0, 5.5)` bites a cap `h = 1.5` deep out of the
/// box's top: the box loses `pi h² (3r - h) / 3`, and the dimple's face has
/// `2 pi r h` of the sphere, both in the exact measure and in the meshes.
#[test]
fn dimple_meshes_its_own_side() {
    let (r, h) = (2.0_f64, 1.5_f64);
    let truth = 1000.0 - PI * h * h * (3.0 * r - h) / 3.0;
    let (topo, piece) = box_less_ball(5.5);
    let volume = solid_volume(&topo, piece, 0.01).unwrap();
    assert!(
        (volume - truth).abs() < 1e-9 * truth,
        "volume {volume}, truth {truth}"
    );
    let mesh = tessellate_solid(&topo, piece, 0.01).unwrap();
    assert!(is_watertight(&mesh), "open or non-manifold mesh");
    let meshed = oriented_solid_volume(&topo, piece, 0.005).unwrap();
    assert!(
        (meshed - truth).abs() < 1e-3 * truth,
        "mesh volume {meshed}, truth {truth}"
    );
    let dimples: Vec<_> = solid_faces(&topo, piece)
        .unwrap()
        .into_iter()
        .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Sphere(_)))
        .collect();
    assert_eq!(dimples.len(), 1, "one sphere face");
    assert!(
        topo.face(dimples[0]).unwrap().is_reversed(),
        "turned inward"
    );
    let cap = 2.0 * PI * r * h;
    for (label, area) in [
        ("exact", face_area(&topo, dimples[0], 0.005).unwrap()),
        ("mesh", mesh_area(&topo, dimples[0])),
    ] {
        assert!(
            (area - cap).abs() < 1e-2 * cap,
            "{label} area {area}, cap {cap}"
        );
    }
}

/// A ball fully inside the box leaves a cavity bounded by its two reversed
/// hemispheres; each hemisphere's own mesh covers its own half, the half its
/// wire leaves on its left about the sphere's outward normal.
#[test]
fn cavity_hemispheres_mesh_their_own_halves() {
    let (topo, piece) = box_less_ball(0.0);
    let volume = solid_volume(&topo, piece, 0.01).unwrap();
    let truth = 1000.0 - 4.0 / 3.0 * PI * 8.0;
    assert!(
        (volume - truth).abs() < 1e-9 * truth,
        "volume {volume}, truth {truth}"
    );
    let mut halves = 0;
    for face in solid_faces(&topo, piece).unwrap() {
        let data = topo.face(face).unwrap();
        if !matches!(data.surface(), FaceSurface::Sphere(_)) {
            continue;
        }
        halves += 1;
        // The wire's turn about +z: counter-clockwise leaves the north half
        // on its left.
        let wire = topo.wire(data.outer_wire()).unwrap();
        let mut turn = 0.0;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge()).unwrap();
            let (a, b) = (
                topo.vertex(edge.start()).unwrap().point(),
                topo.vertex(edge.end()).unwrap().point(),
            );
            let (a, b) = if oe.is_forward() { (a, b) } else { (b, a) };
            turn += a.x().mul_add(b.y(), -(a.y() * b.x()));
        }
        let mesh = tessellate(&topo, face, 0.01).unwrap();
        #[allow(clippy::cast_precision_loss)]
        let mean_z =
            mesh.positions.iter().map(|p| p.z()).sum::<f64>() / mesh.positions.len() as f64;
        assert!(
            mean_z * turn > 0.0,
            "a hemisphere turning {turn} meshed around z {mean_z}"
        );
    }
    assert_eq!(halves, 2, "two hemispheres");
}

/// Scaling the dimpled box by 1.5 along x rebuilds the dimple as the scaled
/// sphere's NURBS image: the volume scales by the same 1.5.
#[test]
fn dimple_survives_an_uneven_scale() {
    let (r, h) = (2.0_f64, 1.5_f64);
    let truth = 1.5 * (1000.0 - PI * h * h * (3.0 * r - h) / 3.0);
    let (mut topo, piece) = box_less_ball(5.5);
    let mut stretch = Mat4::identity();
    stretch.0[0][0] = 1.5;
    transform_solid(&mut topo, piece, &stretch).unwrap();
    let volume = solid_volume(&topo, piece, 0.01).unwrap();
    assert!(
        (volume - truth).abs() < 1e-3 * truth,
        "volume {volume}, truth {truth}"
    );
}

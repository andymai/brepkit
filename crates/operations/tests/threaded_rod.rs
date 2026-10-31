//! A screw thread's ridge, lofted through tooth sections turned about the
//! rod's axis, and fused onto the rod.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_precision_loss)]

use std::f64::consts::TAU;

use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{self, BooleanOp};
use brepkit_operations::{classify, loft, measure, primitives, tessellate, validate};
use brepkit_topology::Topology;
use brepkit_topology::builder::make_planar_face;
use brepkit_topology::explorer::{face_vertices, solid_faces};
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

/// An Acme tooth 1.4 deep, 3.4 wide at its root and 1.8 at its crest, rooted
/// 0.3 inside a rod of radius 7, turned a twentieth of a turn and raised a
/// twentieth of the pitch 4 from one section to the next. Its flanks twist
/// between sections: no plane holds all four corners of one.
fn thread_ridge(topo: &mut Topology, sections: usize) -> SolidId {
    let faces: Vec<_> = (0..=sections)
        .map(|i| {
            let turn = i as f64 * TAU / 20.0;
            let z = 4.0 * turn / TAU;
            let (c, s) = (turn.cos(), turn.sin());
            let at = |u: f64, v: f64| Point3::new((7.0 + u) * c, (7.0 + u) * s, z + v);
            let tooth = [at(-0.3, -1.7), at(1.4, -0.9), at(1.4, 0.9), at(-0.3, 1.7)];
            make_planar_face(topo, &tooth, 1e-7).unwrap()
        })
        .collect();
    loft::loft(topo, &faces).unwrap()
}

/// How far `p` lies from `face`'s surface.
fn off_surface(topo: &Topology, face: brepkit_topology::face::FaceId, p: Point3) -> f64 {
    let surface = topo.face(face).unwrap().surface();
    if let FaceSurface::Plane { normal, d } = surface {
        return (normal.dot(p - Point3::new(0.0, 0.0, 0.0)) - d).abs() / normal.length();
    }
    let (u, v) = surface.project_point(p).unwrap();
    (surface.evaluate(u, v).unwrap() - p).length()
}

/// Every corner of every face of two turns of the ridge lies on its face.
#[test]
fn a_thread_ridges_faces_hold_their_corners() {
    let mut topo = Topology::new();
    let ridge = thread_ridge(&mut topo, 40);
    for face in solid_faces(&topo, ridge).unwrap() {
        for vertex in face_vertices(&topo, face).unwrap() {
            let p = topo.vertex(vertex).unwrap().point();
            let off = off_surface(&topo, face, p);
            assert!(off < 1e-9, "a corner {off} off its face");
        }
    }
    assert!(validate::validate_solid(&topo, ridge).unwrap().is_valid());
}

/// `sections` of the ridge fused onto a rod of radius 7, 20 long from
/// `bottom`: no mesh fallback, a valid solid that meshes closed, and the rod
/// and the ridge's volumes less their overlap, each measured exactly.
fn fuses_exactly(sections: usize, bottom: f64) {
    let mut topo = Topology::new();
    let ridge = thread_ridge(&mut topo, sections);
    let rod = primitives::make_cylinder(&mut topo, 7.0, 20.0).unwrap();
    brepkit_operations::transform::transform_solid(
        &mut topo,
        rod,
        &brepkit_math::mat::Mat4::translation(0.0, 0.0, bottom),
    )
    .unwrap();
    let volume = |topo: &Topology, s| measure::solid_volume(topo, s, 1e-3).unwrap();
    let (v_rod, v_ridge) = (volume(&topo, rod), volume(&topo, ridge));

    let before = boolean::mesh_fallback_count();
    let overlap = boolean::boolean(&mut topo, BooleanOp::Intersect, rod, ridge).unwrap();
    let fused = boolean::boolean(&mut topo, BooleanOp::Fuse, rod, ridge).unwrap();
    assert_eq!(
        boolean::mesh_fallback_count(),
        before,
        "{sections} sections from {bottom}: a boolean fell back"
    );

    assert!(validate::validate_solid(&topo, fused).unwrap().is_valid());
    let mesh = tessellate::tessellate_solid(&topo, fused, 0.01).unwrap();
    assert_eq!(tessellate::boundary_edge_count(&mesh), 0);
    assert_eq!(tessellate::non_manifold_edge_count(&mesh), 0);
    let expected = v_rod + v_ridge - volume(&topo, overlap);
    let measured = volume(&topo, fused);
    assert!(
        (measured - expected).abs() < 1e-6 * expected,
        "{sections} sections from {bottom}: fuse holds {measured}, {expected} expected"
    );
}

/// Two turns of the ridge fused onto a rod that runs past both its ends.
#[test]
fn a_thread_ridge_fuses_onto_its_rod_exactly() {
    fuses_exactly(40, -5.0);
}

/// The ridge starting at the rod's end face, its first sections reaching
/// 1.7 below it. With eight steps the last tooth's lower flank crosses the
/// end face inside its rim, at radius 6.91, and the rim on the flank; with
/// forty the ridge climbs clear of the face.
#[test]
fn a_thread_ridge_through_its_rods_end_fuses_exactly() {
    for sections in [8, 40] {
        fuses_exactly(sections, 0.0);
    }
}

/// Two turns of the ridge put its last section straight above its first,
/// square to the first section's normal. Every face of it faces out: a
/// point just off it along its outward normal lies outside the ridge.
#[test]
fn a_thread_ridge_faces_out() {
    let mut topo = Topology::new();
    let ridge = thread_ridge(&mut topo, 40);
    for face in solid_faces(&topo, ridge).unwrap() {
        let corners = face_vertices(&topo, face).unwrap();
        let mean = corners.iter().fold(Vec3::new(0.0, 0.0, 0.0), |sum, &v| {
            sum + (topo.vertex(v).unwrap().point() - Point3::new(0.0, 0.0, 0.0))
        }) * (1.0 / corners.len() as f64);
        let f = topo.face(face).unwrap();
        let (u, v) = f
            .surface()
            .project_point(Point3::new(0.0, 0.0, 0.0) + mean)
            .unwrap_or((0.0, 0.0));
        let on = match f.surface() {
            FaceSurface::Plane { .. } => Point3::new(0.0, 0.0, 0.0) + mean,
            surface => surface.evaluate(u, v).unwrap(),
        };
        let normal = f.surface().normal(u, v);
        let outward = if f.is_reversed() { -normal } else { normal };
        let off = on + outward.normalize().unwrap() * 0.01;
        assert_eq!(
            classify::classify_point(&topo, ridge, off, 1e-3, 1e-7).unwrap(),
            classify::PointClassification::Outside,
            "a {} face faces in",
            f.surface().type_tag()
        );
    }
}

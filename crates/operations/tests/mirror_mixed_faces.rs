//! A mirror flips handedness. A face with an explicit normal (a plane or a
//! quadric) keeps that normal outward and must reverse its wires; a NURBS
//! face's Su × Sv turns inward, so it flips its flag and keeps its wires. A
//! solid that mixes the two only stays consistent if each follows its own rule.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::Point3;
use brepkit_operations::copy::copy_and_transform_solid;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::{
    boundary_edge_count, non_manifold_edge_count, tessellate_solid,
};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

/// A 2 × 3 × 4 box whose top face is the same plane as a bilinear NURBS
/// patch, with Su × Sv pointing up (out of the box).
fn box_with_nurbs_lid(topo: &mut Topology) -> SolidId {
    let solid = make_box(topo, 2.0, 3.0, 4.0).unwrap();
    let lid = solid_faces(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&f| {
            let face = topo.face(f).unwrap();
            matches!(face.surface(), FaceSurface::Plane { normal, d }
                if normal.z() > 0.5 && (d - 4.0).abs() < 1e-12 && !face.is_reversed())
        })
        .unwrap();
    let patch = NurbsSurface::new(
        1,
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
        vec![
            vec![Point3::new(0.0, 0.0, 4.0), Point3::new(0.0, 3.0, 4.0)],
            vec![Point3::new(2.0, 0.0, 4.0), Point3::new(2.0, 3.0, 4.0)],
        ],
        vec![vec![1.0, 1.0], vec![1.0, 1.0]],
    )
    .unwrap();
    topo.face_mut(lid)
        .unwrap()
        .set_surface(FaceSurface::Nurbs(patch));
    solid
}

fn assert_closed_at_volume(topo: &Topology, solid: SolidId, what: &str) {
    let report = validate_solid(topo, solid).unwrap();
    assert!(report.is_valid(), "{what}: {:?}", report.issues);
    let mesh = tessellate_solid(topo, solid, 0.01).unwrap();
    assert_eq!(boundary_edge_count(&mesh), 0, "{what}: open mesh");
    assert_eq!(
        non_manifold_edge_count(&mesh),
        0,
        "{what}: non-manifold mesh"
    );
    let volume = solid_volume(topo, solid, 0.001).unwrap();
    assert!((volume - 24.0).abs() < 1e-9, "{what}: volume {volume}");
}

#[test]
fn mirrored_mixed_solid_stays_consistent() {
    let mirror = Mat4::translation(1.0, 0.0, 0.0) * Mat4::scale(-1.0, 1.0, 1.0);

    let mut topo = Topology::new();
    let solid = box_with_nurbs_lid(&mut topo);
    assert_closed_at_volume(&topo, solid, "source");
    transform_solid(&mut topo, solid, &mirror).unwrap();
    assert_closed_at_volume(&topo, solid, "transform_solid");

    let mut topo = Topology::new();
    let solid = box_with_nurbs_lid(&mut topo);
    let copy = copy_and_transform_solid(&mut topo, solid, &mirror).unwrap();
    assert_closed_at_volume(&topo, copy, "copy_and_transform_solid");
}

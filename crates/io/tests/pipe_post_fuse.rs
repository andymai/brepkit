//! A post (a cylinder standing on z = 8) fused with a tube lying across it
//! (a cylinder along x from x = -7 to 7, a blind bore in one end) read from
//! STEP. The post is wider than the tube is long, so it stands out past both
//! of the tube's end faces; each end circle of the tube crosses the post's
//! wall at two points, its arc between them a macroscopic distance off the
//! wall. That arc is no section of the wall: taken for one, it split the
//! wall along a line not on it, and the part read back from STEP was wrong.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use brepkit_io::step::read_step;
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::measure::{face_area, oriented_solid_volume, solid_volume};
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology, name: &str) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    read_step(&std::fs::read_to_string(path).unwrap(), topo).unwrap()[0]
}

#[test]
fn a_post_wider_than_a_tube_fuses_with_its_wall_whole() {
    let mut topo = Topology::new();
    let tube = load(&mut topo, "pipe_post_tube.step");
    let post = load(&mut topo, "pipe_post_post.step");
    let before = mesh_fallback_count();
    let res = boolean(&mut topo, BooleanOp::Fuse, tube, post).unwrap();
    assert_eq!(mesh_fallback_count(), before, "mesh fallback");

    // The post's wall stays one face.
    let walls: Vec<_> = solid_faces(&topo, res)
        .unwrap()
        .into_iter()
        .filter(|&f| {
            matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(c) if (c.radius() - 9.0).abs() < 1e-9)
        })
        .collect();
    assert_eq!(walls.len(), 1, "the post's wall split in {}", walls.len());

    // 8089.23 and 647.07 are a reference kernel's volume and wall area for
    // this fuse; the mesh agrees with the exact measure.
    let v = solid_volume(&topo, res, 0.001).unwrap();
    assert!((v - 8089.23).abs() < 0.01, "volume {v}");
    let meshed = oriented_solid_volume(&topo, res, 0.01).unwrap();
    assert!(
        (meshed - v).abs() < 1e-3 * v,
        "meshed volume {meshed}, exact {v}"
    );
    let area = face_area(&topo, walls[0], 0.01).unwrap();
    assert!((area - 647.07).abs() < 0.01, "post wall area {area}");
}

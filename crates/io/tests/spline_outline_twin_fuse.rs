//! Two plates side by side, each extruded from an outline with one spline
//! edge, the two splines one curve: the plates share that wall, and each
//! cap's spline edge has a twin on the other plate with the same two ends.
//!
//! An edge sharing both ends with another along a different path is paved
//! at its middle, and twins on one curve share that middle vertex so they
//! split alike. A twin was left whole when it carried any pave beyond its
//! own two, though the one it carried was at its end: a crossing with
//! another edge there. One twin split and the other did not, the shared
//! wall's edges did not match, and the fuse fell back to a mesh. The
//! operands are a case's, written by brepkit's STEP writer.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use brepkit_io::step::read_step;
use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
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
fn plates_sharing_a_spline_wall_fuse_exactly() {
    let poses = [
        Mat4::identity(),
        Mat4::translation(3.0, -7.0, 2.0) * Mat4::rotation_z(0.7) * Mat4::rotation_x(-1.1),
    ];
    for pose in poses {
        let mut topo = Topology::new();
        let part = load(&mut topo, "spline_outline_twin_part.step");
        let tool = load(&mut topo, "spline_outline_twin_tool.step");
        transform_solid(&mut topo, part, &pose).unwrap();
        transform_solid(&mut topo, tool, &pose).unwrap();
        let apart =
            solid_volume(&topo, part, 1e-3).unwrap() + solid_volume(&topo, tool, 1e-3).unwrap();
        let fused = boolean(&mut topo, BooleanOp::Fuse, part, tool).unwrap();
        assert!(validate_solid(&topo, fused).unwrap().is_valid());
        let faces = solid_faces(&topo, fused).unwrap();
        let count = |pick: fn(&FaceSurface) -> bool| {
            faces
                .iter()
                .filter(|&&f| pick(topo.face(f).unwrap().surface()))
                .count()
        };
        // Each plate's spline and round walls, and the caps, each plate's
        // own beside the other's: not the mesh fallback's facets.
        assert_eq!(count(|s| matches!(s, FaceSurface::Nurbs(_))), 4, "splines");
        assert_eq!(count(|s| matches!(s, FaceSurface::Cylinder(_))), 3, "walls");
        assert_eq!(
            count(|s| matches!(s, FaceSurface::Plane { .. })),
            4,
            "planes"
        );
        // The plates only touch along the wall they share.
        let volume = solid_volume(&topo, fused, 1e-3).unwrap();
        assert!(
            (volume - apart).abs() <= 1e-6 * apart,
            "volume {volume}, want {apart}"
        );
    }
}

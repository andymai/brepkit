//! A part whose flat face lies flush on a tool's cap, the cap bounded by
//! spline edges, fused. The coplanar phase gives each spline edge running
//! inside the partner face as its chord, a section of both faces that
//! partitions the partner. On the face the spline bounds, that chord is the
//! face's own edge and splits nothing; read by its middle alone, which lies
//! off the spline, it was kept, cut the face a second time, and the fuse
//! fell back to a mesh. The operands are a case's, written by brepkit's
//! STEP writer.

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
fn a_part_flush_on_a_spline_capped_tool_fuses_exactly() {
    let poses = [
        Mat4::identity(),
        Mat4::translation(3.0, -7.0, 2.0) * Mat4::rotation_z(0.7) * Mat4::rotation_x(-1.1),
    ];
    for pose in poses {
        let mut topo = Topology::new();
        let part = load(&mut topo, "spline_cap_flush_part.step");
        let tool = load(&mut topo, "spline_cap_flush_tool.step");
        transform_solid(&mut topo, part, &pose).unwrap();
        transform_solid(&mut topo, tool, &pose).unwrap();
        let fused = boolean(&mut topo, BooleanOp::Fuse, part, tool).unwrap();
        assert!(validate_solid(&topo, fused).unwrap().is_valid());
        let faces = solid_faces(&topo, fused).unwrap();
        let count = |pick: fn(&FaceSurface) -> bool| {
            faces
                .iter()
                .filter(|&&f| pick(topo.face(f).unwrap().surface()))
                .count()
        };
        // The tool's four spline sides, the part's wall, and three planes:
        // not the mesh fallback's facets.
        assert_eq!(count(|s| matches!(s, FaceSurface::Nurbs(_))), 4, "splines");
        assert_eq!(count(|s| matches!(s, FaceSurface::Cylinder(_))), 1, "wall");
        assert_eq!(faces.len(), 8, "faces");
        // The reference kernel's result, measured the same way.
        let volume = solid_volume(&topo, fused, 1e-3).unwrap();
        assert!(
            (volume - 12_102.978).abs() <= 1e-4 * 12_102.978,
            "volume {volume}"
        );
    }
}

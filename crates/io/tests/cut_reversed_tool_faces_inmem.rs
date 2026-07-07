//! Guard for cut results whose tool solid carries faces stored with
//! `is_reversed = true`, replayed from the gridfinity tool's literal kernel
//! operands (captured via the arena serializer during a tool probe).
//!
//! The cavity tool here is a compartment pocket whose three side walls are
//! planar NURBS extrusion faces stored reversed (surface normal opposes the
//! effective outward direction). The GFA assembler's Cut step built flipped
//! copies of kept tool faces with `Face::new_reversed` unconditionally — a
//! SET, not a TOGGLE — so an already-reversed tool face came out of the cut
//! unchanged, its effective normal pointing INTO the result material. The
//! B-Rep still paired every edge (the pairing walk is orientation-blind),
//! but the tessellation emitted those walls
//! wound backwards: 12 one-sided mesh edges at every deflection, an STL that
//! slicers flag as non-manifold, and a mesh volume short by the inverted
//! walls' contribution.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::tessellate::{
    TriangleMesh, boundary_edge_count, non_manifold_edge_count, tessellate_solid_with_tolerance,
};
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn load(name: &str, topo: &mut Topology) -> SolidId {
    brepkit_io::arena_io::deserialize_solid(&std::fs::read(fixture(name)).unwrap(), topo).unwrap()
}

/// Signed volume of a triangle mesh (divergence theorem). Positive for an
/// outward-wound closed mesh; inverted face runs subtract their prism terms.
fn signed_mesh_volume(mesh: &TriangleMesh) -> f64 {
    let mut vol6 = 0.0;
    for t in mesh.indices.chunks_exact(3) {
        let p = mesh.positions[t[0] as usize];
        let q = mesh.positions[t[1] as usize];
        let r = mesh.positions[t[2] as usize];
        vol6 += p.x() * (q.y() * r.z() - q.z() * r.y()) - p.y() * (q.x() * r.z() - q.z() * r.x())
            + p.z() * (q.x() * r.y() - q.y() * r.x());
    }
    vol6 / 6.0
}

const QUALITY_TIERS: [(f64, f64); 3] = [(0.01, 5.0), (0.1, 12.0), (0.5, 15.0)];

#[test]
fn cut_with_reversed_tool_faces_is_watertight_and_volume_consistent() {
    let mut topo = Topology::new();
    let body = load("compart_nurbswall_cavity_body.bin", &mut topo);
    let tool = load("compart_nurbswall_cavity_tool.bin", &mut topo);

    // vol(cut) = vol(body) − vol(body ∩ tool). The cavity pokes ~1mm above
    // the body's top face, so vol(tool) alone would over-count the removed
    // material; measure the common region with an Intersect instead. Both
    // sides come from meshes of the same pipeline, so the check is
    // self-consistent (no magic constants).
    let mesh_body = tessellate_solid_with_tolerance(&topo, body, 0.05, 0.1).unwrap();
    let common = boolean(&mut topo, BooleanOp::Intersect, body, tool).unwrap();
    let mesh_common = tessellate_solid_with_tolerance(&topo, common, 0.05, 0.1).unwrap();
    let expected = signed_mesh_volume(&mesh_body) - signed_mesh_volume(&mesh_common);

    let result = boolean(&mut topo, BooleanOp::Cut, body, tool).unwrap();

    for (deflection, angular_deg) in QUALITY_TIERS {
        let mesh =
            tessellate_solid_with_tolerance(&topo, result, deflection, angular_deg.to_radians())
                .unwrap();
        assert_eq!(
            boundary_edge_count(&mesh),
            0,
            "one-sided mesh edges at deflection {deflection}"
        );
        assert_eq!(
            non_manifold_edge_count(&mesh),
            0,
            "non-manifold mesh edges at deflection {deflection}"
        );
        let vol = signed_mesh_volume(&mesh);
        let err = (vol - expected).abs() / expected.abs();
        assert!(
            err < 0.02,
            "mesh volume {vol:.1} deviates {:.1}% from expected {expected:.1} \
             at deflection {deflection} — inverted faces?",
            err * 100.0
        );
    }
}

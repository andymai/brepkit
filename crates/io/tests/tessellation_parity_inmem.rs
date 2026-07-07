//! Tessellation-parity guards for boolean-result faces, replayed from the
//! gridfinity tool's literal kernel operands (captured via the arena
//! serializer during a tool probe).
//!
//! Two defects are pinned here:
//!
//! 1. **Ruled-direction interior grid over-mesh.** `interior_grid_resolution`
//!    fed a cylinder/cone face's `dv` — the axial span in millimeters — to the
//!    chord-deviation formula, which expects an arc angle in radians. A 28 mm
//!    tall hex-cut corner cylinder meshed at ~7700 triangles (one interior row
//!    per angular-tolerance step of "arc"), blowing a 2×2×4 honeycomb bin to
//!    ~63k triangles where ~4k is expected. The v direction of a developable
//!    band has zero chord sag; two interior rows suffice.
//!
//! 2. **Partial-band rim cracks at fine deflection.** Non-full-revolution
//!    cylinder/cone bands (the gridfinity socket-profile corner rings) fell to
//!    the snap mesher, which re-samples the rim independently and reconciles
//!    by 1e-6 proximity. At export tolerance (0.01 mm) its segment count
//!    diverges from the shared edge pool's, leaving ~200 one-sided mesh edges
//!    on the compartment cavity cut (non-watertight STL). Hole-free partial
//!    bands now triangulate via CDT over the shared pool ids, which is
//!    watertight by construction.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::tessellate::{
    boundary_edge_count, non_manifold_edge_count, tessellate_solid_grouped_with_tolerance,
    tessellate_solid_with_tolerance,
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

/// The tool's tessellation quality tiers (export, preview-with-lip, coarse).
const QUALITY_TIERS: [(f64, f64); 3] = [(0.01, 5.0), (0.03, 8.0), (0.5, 15.0)];

#[test]
fn compartment_cavity_cut_tessellates_watertight_at_all_quality_tiers() {
    let mut topo = Topology::new();
    let body = load("compart_cavity_cut_body.bin", &mut topo);
    let tool = load("compart_cavity_cut_tool.bin", &mut topo);
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
    }
}

#[test]
fn hex_cut_body_tessellation_density_is_bounded() {
    let mut topo = Topology::new();
    let body = load("honeycomb_hexcut_body.bin", &mut topo);
    let tool = load("honeycomb_hexcut_tool.bin", &mut topo);
    let result = boolean(&mut topo, BooleanOp::Cut, body, tool).unwrap();

    // The tool's preview tier for lip bins.
    let (mesh, face_offsets) =
        tessellate_solid_grouped_with_tolerance(&topo, result, 0.03, 8.0_f64.to_radians()).unwrap();
    assert_eq!(boundary_edge_count(&mesh), 0, "one-sided mesh edges");
    assert_eq!(non_manifold_edge_count(&mesh), 0, "non-manifold mesh edges");

    let tris = mesh.indices.len() / 3;
    // Pre-fix this body meshed at 14272 triangles (two corner cylinders at
    // 7716 and 4892 alone); post-fix it is ~2100. The bound leaves headroom
    // for sampling changes while catching any order-of-magnitude regression.
    assert!(tris < 5000, "tessellation too dense: {tris} triangles");

    let max_face_tris = face_offsets
        .windows(2)
        .map(|w| (w[1] - w[0]) / 3)
        .max()
        .unwrap_or(0);
    assert!(
        max_face_tris < 1000,
        "a single face meshed at {max_face_tris} triangles"
    );
}

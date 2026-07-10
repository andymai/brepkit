//! Faithful guard for the mesh-boolean fallback on coincident-wall contact.
//!
//! The dovetail relief cut (tongue prism minus tapered socket pocket) makes
//! the GFA boolean fail its assembly gate ("all shells classified as
//! holes"), so `boolean()` falls back to the co-refinement mesh boolean.
//! The operands touch with exactly coincident walls; the pre-fix splitter
//! left T-junctions between co-refined triangles and reduced coplanar
//! contact to a single longest segment, and the winding-number classifier
//! coin-flipped on the shared walls (winding is exactly 1/2 there) — the
//! fallback exported open, non-manifold meshes (raw bnd=11; export bnd=15
//! nm=1 on this pair, nm=76 on an integer-width socket fuse).
//!
//! The conforming rewrite (CDT re-triangulation with cross-triangle
//! edge-point propagation, mutual coplanar edge clipping, and explicit
//! OnSame/OnOpp coincident-surface classification) must keep the fallback
//! output closed and 2-manifold, and the result now self-reports welded
//! boundary / non-manifold edge counts so `boolean()` can see a bad result.
//!
//! Fixtures are the tool's EXACT serialized operands: `relief_tongue.bin`
//! (6-face trapezoid tongue prism) and `relief_cutter.bin` (tapered socket
//! pocket, 18 faces with cones and cylinders).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::mesh_boolean::mesh_boolean;
use brepkit_operations::tessellate::{
    TriangleMesh, tessellate_solid_for_boolean, tessellate_solid_with_tolerance,
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

fn mesh_health(mesh: &TriangleMesh, grid: f64) -> (usize, usize) {
    type Q = (i64, i64, i64);
    let s = 1.0 / grid;
    let q = |p: Point3| -> Q {
        (
            (p.x() * s).round() as i64,
            (p.y() * s).round() as i64,
            (p.z() * s).round() as i64,
        )
    };
    let mut occ: HashMap<(Q, Q), usize> = HashMap::new();
    for tri in mesh.indices.chunks_exact(3) {
        let a = q(mesh.positions[tri[0] as usize]);
        let b = q(mesh.positions[tri[1] as usize]);
        let c = q(mesh.positions[tri[2] as usize]);
        if a == b || b == c || a == c {
            continue;
        }
        for (p, r) in [(a, b), (b, c), (c, a)] {
            let key = if p <= r { (p, r) } else { (r, p) };
            *occ.entry(key).or_default() += 1;
        }
    }
    let bnd = occ.values().filter(|&&c| c == 1).count();
    let nm = occ.values().filter(|&&c| c > 2).count();
    (bnd, nm)
}

fn mesh_signed_volume(mesh: &TriangleMesh) -> f64 {
    let origin = Point3::new(0.0, 0.0, 0.0);
    let mut vol = 0.0;
    for tri in mesh.indices.chunks_exact(3) {
        let a = mesh.positions[tri[0] as usize] - origin;
        let b = mesh.positions[tri[1] as usize] - origin;
        let c = mesh.positions[tri[2] as usize] - origin;
        vol += a.dot(b.cross(c)) / 6.0;
    }
    vol
}

#[test]
fn relief_cut_raw_mesh_boolean_is_manifold() {
    let mut topo = Topology::new();
    let tongue = load("relief_tongue.bin", &mut topo);
    let cutter = load("relief_cutter.bin", &mut topo);

    let mesh_a = tessellate_solid_for_boolean(&topo, tongue, 0.1, 0.0).unwrap();
    let mesh_b = tessellate_solid_for_boolean(&topo, cutter, 0.1, 0.0).unwrap();
    for (label, m) in [("tongue", &mesh_a), ("cutter", &mesh_b)] {
        let (bnd, nm) = mesh_health(m, 1e-4);
        assert_eq!((bnd, nm), (0, 0), "operand {label} must be watertight");
    }
    let tongue_vol = mesh_signed_volume(&mesh_a);

    let result = mesh_boolean(&mesh_a, &mesh_b, BooleanOp::Cut, 1e-7).unwrap();
    assert_eq!(
        result.boundary_edge_count, 0,
        "self-reported boundary edges on the relief cut"
    );
    assert_eq!(
        result.non_manifold_edge_count, 0,
        "self-reported non-manifold edges on the relief cut"
    );
    for grid in [1e-3, 1e-4, 1e-5, 1e-6] {
        let (bnd, nm) = mesh_health(&result.mesh, grid);
        assert_eq!(
            (bnd, nm),
            (0, 0),
            "relief cut mesh must be closed and manifold at grid {grid:e}"
        );
    }
    let vol = mesh_signed_volume(&result.mesh);
    assert!(
        vol > 0.0 && vol < tongue_vol,
        "relief cut volume must be positive and below the tongue's ({vol} vs {tongue_vol})"
    );
}

#[test]
fn relief_cut_boolean_fallback_export_is_manifold() {
    let mut topo = Topology::new();
    let tongue = load("relief_tongue.bin", &mut topo);
    let cutter = load("relief_cutter.bin", &mut topo);

    let result = boolean(&mut topo, BooleanOp::Cut, tongue, cutter).unwrap();
    let mesh = tessellate_solid_with_tolerance(&topo, result, 0.01, 5f64.to_radians()).unwrap();
    for grid in [1e-3, 1e-4, 1e-5] {
        let (bnd, nm) = mesh_health(&mesh, grid);
        assert_eq!(
            (bnd, nm),
            (0, 0),
            "relief cut export mesh must be closed and manifold at grid {grid:e}"
        );
    }
}

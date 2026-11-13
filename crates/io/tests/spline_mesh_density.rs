//! The mesh of a spline surface follows the surface: operands captured from
//! brepjs's `turbofan-engine` playground example.
//!
//! `spline_nacelle.bin` is a closed spline profile revolved one full turn,
//! 260 across: 16 NURBS faces. Its profile runs almost straight for most of
//! its length and turns tight at the trailing lip. `lofted_stator_blade.bin`
//! is one lofted aerofoil blade: two NURBS faces with a tight leading edge.
//! A mesh divided evenly takes the tight place's spacing everywhere.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{
    TriangleMesh, boundary_edge_count, non_manifold_edge_count, tessellate_solid_with_tolerance,
};
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology, name: &str) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    deserialize_solid(&std::fs::read(path).unwrap(), topo).unwrap()
}

fn enclosed(mesh: &TriangleMesh) -> f64 {
    mesh.indices
        .chunks_exact(3)
        .map(|t| {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| mesh.positions[i as usize]);
            let v =
                |p: brepkit_math::vec::Point3| brepkit_math::vec::Vec3::new(p.x(), p.y(), p.z());
            v(a).dot(v(b).cross(v(c))) / 6.0
        })
        .sum()
}

/// Meshes `name` and checks the mesh is closed, holds the solid's volume
/// within `within` of it, and takes at most `most` triangles.
fn assert_lean(name: &str, deflection: f64, most: usize, within: f64) {
    let mut topo = Topology::new();
    let solid = load(&mut topo, name);
    let mesh = tessellate_solid_with_tolerance(&topo, solid, deflection, 0.3).unwrap();
    assert_eq!(boundary_edge_count(&mesh), 0, "{name}: open mesh");
    assert_eq!(non_manifold_edge_count(&mesh), 0, "{name}: over-used edges");
    let triangles = mesh.indices.len() / 3;
    assert!(triangles <= most, "{name}: {triangles} triangles");
    let truth = solid_volume(&topo, solid, 0.001).unwrap();
    let volume = enclosed(&mesh);
    assert!(
        (volume - truth).abs() <= within * truth,
        "{name}: mesh volume {volume}, solid {truth}"
    );
}

/// An even mesh of the nacelle at this tolerance took 211,504 triangles.
#[test]
fn a_revolved_spline_meshes_by_its_own_curvature() {
    assert_lean("spline_nacelle.bin", 0.1, 30_000, 2e-3);
}

/// An even mesh of the blade at this tolerance took 1,228 triangles, and
/// left two edges over-used.
#[test]
fn a_lofted_blade_meshes_by_its_own_curvature() {
    assert_lean("lofted_stator_blade.bin", 0.1, 300, 2e-2);
}

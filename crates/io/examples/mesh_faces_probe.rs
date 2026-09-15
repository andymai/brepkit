//! Per-face mesh census of one serialized solid: surface type, orientation
//! flag, triangle count and signed divergence flux at a chosen tessellation,
//! so two mesher configurations can be diffed face by face.
//!
//! ```text
//! A=<solid.bin> DEFL=0.01 ANG=5 cargo run --release -p brepkit-io --example mesh_faces_probe
//! ```
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout)]

use brepkit_math::vec::Vec3;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;

fn main() {
    let path = std::env::var_os("A").expect("A=<path>");
    let defl: f64 = std::env::var("DEFL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.01);
    let ang: f64 = std::env::var("ANG")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5.0);
    let mut topo = Topology::new();
    let sid =
        brepkit_io::arena_io::deserialize_solid(&std::fs::read(path).unwrap(), &mut topo).unwrap();
    let faces = solid_faces(&topo, sid).unwrap();
    let (mesh, offsets) = brepkit_operations::tessellate::tessellate_solid_grouped_with_tolerance(
        &topo,
        sid,
        defl,
        ang.to_radians(),
    )
    .unwrap();
    let mut total = 0.0;
    for (i, fid) in faces.iter().enumerate() {
        let face = topo.face(*fid).unwrap();
        let (from, to) = (offsets[i] as usize, offsets[i + 1] as usize);
        let flux: f64 = mesh.indices[from..to]
            .chunks_exact(3)
            .map(|t| {
                let p = |k: usize| {
                    let q = mesh.positions[t[k] as usize];
                    Vec3::new(q.x(), q.y(), q.z())
                };
                p(0).dot(p(1).cross(p(2))) / 6.0
            })
            .sum();
        total += flux;
        println!(
            "face {fid:?} {} rev={} tris={} flux={flux:.4}",
            face.surface().type_tag(),
            face.is_reversed(),
            (to - from) / 3
        );
    }
    println!(
        "total flux={total:.3} bnd={} nm={}",
        brepkit_operations::tessellate::boundary_edge_count(&mesh),
        brepkit_operations::tessellate::non_manifold_edge_count(&mesh)
    );
}

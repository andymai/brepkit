//! Profile export-tolerance tessellation on a captured tool solid (gh #1500).
//!
//! Usage: cargo run --release -p brepkit-io --example profile_export_tess -- <solid.bin> [deflection] [angular]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout)]

use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map_or("/tmp/perfbench/export-bin.bin", |s| s);
    let deflection: f64 = args.get(2).map_or(0.01, |s| s.parse().unwrap());
    let angular: f64 = args.get(3).map_or(5.0, |s| s.parse().unwrap());

    let bytes = std::fs::read(path).unwrap();
    let mut topo = brepkit_topology::Topology::new();
    let solid = brepkit_io::arena_io::deserialize_solid(&bytes, &mut topo).unwrap();

    let counts = brepkit_topology::explorer::solid_entity_counts(&topo, solid).unwrap();
    println!(
        "solid: faces={} edges={} vertices={}",
        counts.0, counts.1, counts.2
    );

    for round in 0..3 {
        let t = Instant::now();
        let (mesh, offsets) =
            brepkit_operations::tessellate::tessellate_solid_grouped_with_tolerance(
                &topo, solid, deflection, angular,
            )
            .unwrap();
        let mut h: u64 = 0;
        for &i in &mesh.indices {
            h = h.wrapping_mul(31).wrapping_add(u64::from(i));
        }
        for p in &mesh.positions {
            for c in [p.x(), p.y(), p.z()] {
                h = h.wrapping_mul(31).wrapping_add(c.to_bits());
            }
        }
        println!(
            "round {round}: {:.1}ms tris={} verts={} groups={} hash={h:x}",
            t.elapsed().as_secs_f64() * 1e3,
            mesh.indices.len() / 3,
            mesh.positions.len(),
            offsets.len()
        );
    }
}
